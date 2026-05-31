# LocateAnything_object_detection_with_probability_windows
Implement of LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding in windows system, and we also extend the inference result with object detection confidence. [https://github.com/NVlabs/Eagle/tree/main/Embodied]   

<img width="2752" height="1536" alt="Vision-Language_Object_Grounding_Framework" src="https://github.com/user-attachments/assets/ad91d996-6504-44f0-8ac3-791f4892a05a" />

## 1.windows environment setting up
1. pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
2. pip install deepspeed==0.16.4 --only-binary=:all:  
3. pip install transformers==4.57.1 tokenizers==0.22.0 sentencepiece==0.2.0 shortuuid accelerate==1.5.2 peft==0.12.0 bitsandbytes "wandb>=0.17.7" pydantic==2.7.1 "markdown2[all]" "numpy<2,>=1.25" "scikit-learn>=1.2.2" gradio==3.35.2 gradio_client==0.2.9 lmdb requests httpx==0.24.0 uvicorn fastapi streamlit streamlit-image-select einops einops-exts "timm>=1.0.11"  
4. pip install -e . (ignore pip installing failed)  
5. pip install opencv-python  
6. pip install decord   
7. According to the guide in cuda12.8 windows system.Then the final list is:
<img width="1848" height="1016" alt="image" src="https://github.com/user-attachments/assets/17489db6-8906-4d3d-b2ff-b47c0ba6de46" />
<img width="1854" height="1008" alt="image" src="https://github.com/user-attachments/assets/5baed159-0b59-46cf-ab75-a321ff4c6f1f" />
<img width="1858" height="1006" alt="image" src="https://github.com/user-attachments/assets/455d7e38-a632-41b9-8be2-0629d8330cad" />
<img width="1850" height="1014" alt="image" src="https://github.com/user-attachments/assets/f2c9ad07-f222-4f06-bb5d-6ecb8cc125d1" />
<img width="1850" height="1012" alt="image" src="https://github.com/user-attachments/assets/17143e71-5673-450e-99f2-324de00c5f01" />

## 2. Improment idea to achieve object detection with probability in LocateAnything
As a multimodal large language model, LocateAnything generates output through autoregressive text sequences (<box><x1><y1><x2><y2></box>), rather than directly outputting bounding box coordinates and classification logits from a neural network like traditional detectors such as YOLO. Therefore, we leverage the softmax probability that the language model assigns to each token during generation to reflect the model's "degree of certainty" in selecting that token, and the geometric mean of the probabilities of all tokens constituting a detection box is used as the overall confidence score.  
So the Core Idea is: When generating each token, the model outputs a logits vector (scoring all possible tokens), which is then converted into a probability distribution via softmax. The probability value assigned to the selected token within this distribution reflects how "certain" the model is about that choice.  
1. Modify the locateanything_worker.py as we proposed;
2. Demo the Location with: python detect_and_draw.py bus.jpg person,car,bicycle,bus
<img width="1850" height="1250" alt="image" src="https://github.com/user-attachments/assets/a19331e1-1a96-462f-8df0-7a44af5f10cb" />
We found the memory cost almost 13GB

## 3. windows inference results
The result is as shown:
<img width="3098" height="1644" alt="image" src="https://github.com/user-attachments/assets/0d785ad2-a043-44d2-9f55-aba8d3b0e087" />
And the inference result is saved in output_bus.jpg.

## 4. Others
we can use pybind11 to implement C++ inference, and we also can quantization the model. Also the official demo, such as Phrase Grounding, Scene Text Detection and so on.

## Thanks
Thanks for their extremelly perfect work.     

@article{wang2025locateanything,
  title   = {LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding},
  author  = {Shihao Wang and Shilong Liu and Yuanguo Kuang and Xinyu Wei and
             Yangzhou Liu and Zhiqi Li and Yunze Man and Guo Chen and
             Andrew Tao and Guilin Liu and Jan Kautz and Lei Zhang and Zhiding Yu},
  journal = {arXiv:2605.27365},
  year    = {2026},
}
