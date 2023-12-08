import os
import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import open_clip

from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, TextStreamer

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

class LLaVA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        disable_torch_init()

        model_name = get_model_name_from_path(cfg.TRAINER.LLM.LLAVA)

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(cfg.TRAINER.LLM.LLAVA, None, model_name, False, False, device="cuda")
        
        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        
        self.conv = conv_templates[conv_mode].copy()
        if "mpt" in model_name.lower():
            self.roles = ('user', 'assistant')
        else:
            self.roles = self.conv.roles

        self.image_tensor = None

    def forward(self, image_path, text, system=None, temperature=0.2, max_new_tokens=1024):
        inp = text
        self.conv.system = system if system != None else self.conv.system
        image = load_image(image_path) if image_path != None else None
        import pdb; pdb.set_trace()
        if image is not None:
            image_tensor = process_images([image], self.image_processor, self.model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

            self.image_tensor = image_tensor
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

        else:
            image_tensor = self.image_tensor


        self.conv.append_message(self.conv.roles[0], inp)
        
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        # streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # with torch.inference_mode():
        #     output_ids = self.model.generate(
        #         input_ids,
        #         images=image_tensor,
        #         do_sample=True if temperature > 0 else False,
        #         temperature=temperature,
        #         max_new_tokens=max_new_tokens,
        #         streamer=streamer,
        #         use_cache=True,
        #         stopping_criteria=[stopping_criteria])

        # outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        # self.conv.messages[-1][-1] = outputs

        # return outputs
        streamer = TextIteratorStreamer(
            self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
        )

        generate_kwargs = dict(
            {"input_ids": input_ids},
            images=image_tensor,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )
        with torch.inference_mode():
            t = Thread(target=self.model.generate, kwargs=generate_kwargs)
            t.start()

        outputs = []
        for text in streamer:
            outputs.append(text)

        self.conv.messages[-1][-1] = "".join(outputs)
        return "".join(outputs)

        

class LLaMA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.TRAINER.LLM.LLAMA)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.TRAINER.LLM.LLAMA,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer.use_default_system_prompt = False

        self.MAX_MAX_NEW_TOKENS = 2048
        self.DEFAULT_MAX_NEW_TOKENS = 1024
        self.MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

    def forward(
        self,
        message: str,
        chat_history: list[tuple[str, str]],
        system_prompt: str = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
    ) -> str:
        conversation = []

        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})

        for user, assistant in chat_history:
            conversation.extend(
                [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            )
        conversation.append({"role": "user", "content": message})

        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt")
        if input_ids.shape[1] > self.MAX_INPUT_TOKEN_LENGTH:
            input_ids = input_ids[:, -self.MAX_INPUT_TOKEN_LENGTH:]

        input_ids = input_ids.to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
        )

        generate_kwargs = dict(
            {"input_ids": input_ids},
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=1,
            repetition_penalty=repetition_penalty,
        )

        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        outputs = []
        for text in streamer:
            outputs.append(text)

        return "".join(outputs)

@TRAINER_REGISTRY.register()
class LLM(TrainerX):
    def build_model(self):
        cfg = self.cfg      

        # load clip model
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir=cfg.TRAINER.LLM.CLIP)
        self.img_tokenizer = open_clip.get_tokenizer('ViT-B-32')

        # load llama model
        self.llama = LLaMA(cfg)
        
        # load llava model
        self.llava = LLaVA(cfg)

        self.optim = build_optimizer(self.llama, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)  

        self.register_model("LLM", self.llama, self.optim, self.sched)

        self.llama_sys_prompt = f"Given a list of labels from the {self.cfg.DATASET.NAME} dataset, generate a unique and descriptive prompt for an image that represents each label. For each label, create one prompt only. Format the output as a list, with each prompt starting with a dash. Focus on capturing the unique characteristics of each pet breed or category represented by the labels, ensuring diversity and accuracy in your descriptions."
        self.llama_begin_prompt = lambda x : f'''I have a list of labels from the {self.cfg.DATASET.NAME} datases: {x} For each label, I need a creative and descriptive prompt that could be used to generate an image representing that label. Please provide one prompt for each label, formatted as a list with each prompt starting with a dash. Ensure the prompts are diverse and accurately reflect the essence of each label.'''
        self.llama_middle_prompt = "Please help me refine the prompt based on this label and the following suggestions, ultimately generating one high-quality prompts. Each output should occupy one line, starting with a '-' as the first character."
        self.llama_chat_his = []
        self.llava_begin_prompt = lambda x, y : f"You are given an image, along with several related prompts for this image, as well as scores for each of these prompts. Please help me by combining the image and the scores for the prompts to provide suggestions for each prompt. My list of prompts is: {x}, and the list of scores is: {y}. Each line of the output should start with a '-', first listing the original prompt, followed by 'Suggest is', and then your suggestion."

        
    def forward_backward(self, batch):
        classname = batch['classname'][0]

        response1 = self.llama(self.llama_begin_prompt(self.dm.dataset.classnames), self.llama_chat_his, self.llama_sys_prompt)
        self.llama_chat_his.append((self.llama_begin_prompt(self.dm.dataset.classnames), response1))

        prompts = [line.split('-')[1].strip() for line in response1.split('\n') if line.strip().startswith('-')]

        image = self.preprocess(Image.open(batch["impath"][0])).unsqueeze(0)
        text = self.img_tokenizer(prompts)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        llava_response = self.llava(batch["impath"][0], "I have an image that I've tested against a set of prompts using CLIP to get matching scores. First, please describe the image in detail.", "Analyze and describe the provided image in detail, focusing on its prominent features, context, and any identifiable elements.")
        print(llava_response)
        import pdb; pdb.set_trace()
        sys_prompt = "Once the image description is provided, you will receive a list of prompts with their corresponding CLIP scores and the image's true label. Your task is to evaluate the prompt that corresponds to the true label of the image. Consider the image's actual content and characteristics to suggest how the prompt can be improved for a more accurate and descriptive representation of the image."
        prompt = "After your description, I will provide you with a list of prompts and their corresponding scores from the CLIP test, along with the real label of the image. I need you to review these and provide improvement suggestions for the prompt that matches the real label of the image, based on its content and characteristics."
        prompt1 = f"The list of prompts is {prompts} and the corresponding CLIP scores is {text_probs}. The truth label is {classname}"
        import pdb; pdb.set_trace()
        llava_response = self.llava(None, f"{prompt} {prompt1}", sys_prompt)

        response2 = self.llama(self.llama_middle_prompt + llava_response, self.llama_chat_his, self.llama_sys_prompt)
        prompts2 = [line.split('*')[1].strip() for line in response1.split('\n') if line.strip().startswith('*')]

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
