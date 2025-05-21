[VisualCloze](https://github.com/lzyhha/VisualCloze): A Universal Image Generation Framework via Visual In-Context Learning

# Tips
* 量化fp8的模式始终跑不出效果，官方的diffuser版本暂时没空捣鼓，先放代码出来，免得说占坑。方法需要的显存较大，内存也要64+,24显存可以试试关闭cpu offload

1.Installation  
-----
  In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_VisualCloze.git
```  
  
2.requirements  
----
```
pip install -r requirements.txt
```

3 models 
----
[lora 384](https://huggingface.co/VisualCloze/VisualClozePipeline-LoRA-384/tree/main) or [lora 512](https://huggingface.co/VisualCloze/VisualClozePipeline-LoRA-512/tree/main)
```
├── ComfyUI/models/loras/
|             ├── visualcloze-lora-512.safetensors # or 384
├── ComfyUI/models/diffusion_models/
|             ├── flux1-fill-dev.safetensors # or flux1-fill-dev-fp8.safetensors
```

# Example
![](https://github.com/smthemex/ComfyUI_VisualCloze/blob/main/add.png)
![](https://github.com/smthemex/ComfyUI_VisualCloze/blob/main/exampleA.png)
![](https://github.com/smthemex/ComfyUI_VisualCloze/blob/main/VisualCloze.png)

# Citation
```
@article{li2025visualcloze,
  title={VisualCloze : A Universal Image Generation Framework via Visual In-Context Learning},
  author={Li, Zhong-Yu and Du, Ruoyi and Yan, Juncheng and Zhuo, Le and Li, Zhen and Gao, Peng and Ma, Zhanyu and Cheng, Ming-Ming},
  journal={arXiv preprint arxiv:},
  year={2025}
}
```
