# AI Agent as any Artist

## 📝 Abstract

- **AI application** to draw any painting like human, who can use a small number of strokes to create masterpiece paintings.

- Experiments demonstrate that masterpiece **visual quality** can be achieved using only hundreds of strokes. 

- The training process does not require any ~~supervised data~~. 

## 🎬 Demo

### Paint Learner with DDPG (Deep Deterministic Policy Gradient) + SBR (Stroke-based Rendering)

![Sample](./samples/monalisa.png) 🠊 ![Result](./assets/PaintDDPG/monalisa.gif)

<details>
  <summary>More samples</summary>
  <img src="./samples/Fernando-Botero-Mona-Lisa.gif" width=256>
  <img src="./samples/MidJourney-1er-prix-Colorado-State-Fair.gif" width=256>
  <img src="./samples/pablo-picasso-mediterranean-landscape.gif" width=256>
  <img src="./samples/pablo-picasso-the-dream.gif" width=256>
  <img src="./samples/salvador-dali-combined-paintings.gif" width=256>
  <img src="./samples/salvador-dali-Galatea-of-the-Spheres.gif" width=256>
  <img src="./samples/van-gogh-garden-at-arles.gif" width=256>
  <img src="./samples/van-gogh-starry-night.gif" width=256>
</details>

<details>
  <summary>Much More samples</summary>
  <img src="./assets/PaintDDPG/sunrise.gif" width=256>
  <img src="./assets/PaintDDPG/sunflower.gif" width=256>
  <img src="./assets/PaintDDPG/palacemuseum.gif" width=256>
  <img src="./assets/PaintDDPG/deepdream_night.gif" width=256>
  <img src="./assets/PaintDDPG/deepdream_bird.gif" width=256>
</details>

### Paint Transformer

<img src="./assets/PaintTransformer/1x2.gif" width=512>

<details>
  <summary>More samples</summary>
  <img src="./assets/PaintTransformer/2x2.gif" width=512>
  <img src="./assets/PaintTransformer/3x2.gif" width=512>
</details>

## 🔢 Model Weights

### Paint Learner with DDPG + SBR

🖼️ Resolution of model output is **128** x **128**. 

📱 For hires image, input will be divided into **patch**es and refined in parallel. But, it will cost more computation and storage.

- **default** stroke: [renderer.pkl](https://drive.google.com/open?id=1-7dVdjCIZIxh8hHJnGTK-RA1-jL1tor4) and [actor.pkl](https://drive.google.com/open?id=1a3vpKgjCVXHON4P7wodqhCgCMPgg1KeR)

- **triangle** stroke: [renderer.pkl](https://drive.google.com/open?id=1YefdnTuKlvowCCo1zxHTwVJ2GlBme_eE) and [actor.pkl](https://drive.google.com/open?id=1k8cgh3tF7hKFk-IOZrgsUwlTVE3CbcPF)

- **circle** stroke: [renderer.pkl](https://drive.google.com/open?id=1kI4yXQ7IrNTfjFs2VL7IBBL_JJwkW6rl) and [actor.pkl](https://drive.google.com/open?id=1ewDErUhPeGsEcH8E5a2QAcUBECeaUTZe)

- **curve** stroke: [renderer.pkl](https://drive.google.com/open?id=1XUdti00mPRh1-1iU66Uqg4qyMKk4OL19) and [actor.pkl](https://drive.google.com/open?id=1VBtesw2rHmYu2AeJ22XvTCuzuqkY8hZh)

### Paint Transformer

🖼️ Resolution of model output is **32** x **32**. 

🔀 For hires image (size of $2^N$), input will be divided into **patch**es at each k = 0 .. $log{_2}{(N/32)}+1$, then, all patches at each k are processed in parallel.

- Checkpoint: [Paddle](https://drive.google.com/file/d/1G0O81qSvGp0kFCgyaQHmPygbVHFi1--q/view?usp=sharing) / [PyTorch](https://drive.google.com/file/d/1NDD54BLligyr8tzo8QGI5eihZisXK1nq/view?usp=sharing)

- Primitive brushes: 
  - **horizontal** <img src="./checkpoints/paint_transformer/brush/brush_large_horizontal.png" width=50>
  - **vertical** <img src="./checkpoints/paint_transformer/brush/brush_large_vertical.png" width=50>
  - **rendering** (diffentiable): <img src="./assets/PaintTransformer/metabrush.png" width=250>

## 🦾 Features 🦿

✅ Code Restructure

✅ Patch Processing for higher resolution

✅ Brushstrokes

  - ✅ Triangle

  - ✅ Circle

  - ✅ Curve ([Bézier](https://pomax.github.io/bezierinfo/))
    
    <img src="./assets/bezier_curve_quadratic.png">

  - ✅ Template

⬜ Simple GUI

⬜ Motion Path with Drawing Hand ([video](https://youtu.be/QQyag99u1sk))

⬜ Tool Control with advanced Brushes

  - ⬜ MS Paint 

    <img src="./assets/brushes_mspaint.png" height=250>

⬜ Real-world simulation (with **robotic arm**)

## 🔎 Reference

1. Learning to Paint with **Model-based Deep Reinforcement Learning** ([Paper](https://arxiv.org/pdf/1903.04411) | [Code](https://github.com/hzwer/ICCV2019-LearningToPaint) | [Demo](https://replicate.com/hzwer/iccv2019-learningtopaint)) 
<img src="./assets/PaintDDPG/poster.png">

2. **Paint Transformer**: Feed-Forward Neural Painting with *Stroke Prediction* ([Paper](https://arxiv.org/abs/2108.03798) | [Code/Paddle](https://github.com/wzmsltw/PaintTransformer) | [Code/Torch](https://github.com/Huage001/PaintTransformer) | [Demo](https://huggingface.co/spaces/akhaliq/PaintTransformer))
<img src="./assets/PaintTransformer/abstract.png">

<details>

  <summary>Style Transfer</summary>

  3. Learning to Generate **Line Drawings** that convey *Geometry* and *Semantics* ([Paper](https://arxiv.org/abs/2203.12691) | [Code](https://github.com/carolineec/informative-drawings) | [Demo](https://github.com/carolineec/informative-drawings))

  4. CSGO: **Content-Style Composition** in Text-to-Image Generation ([Paper](https://arxiv.org/abs/2408.16766) | [Code](https://github.com/instantX-research/CSGO) | [Demo](https://huggingface.co/spaces/xingpng/CSGO))

  5. Rethinking Style Transfer: From Pixels to **Parameterized Brushstrokes** ([Paper](https://arxiv.org/abs/2103.17185) | [Code](https://github.com/CompVis/brushstroke-parameterized-style-transfer))

</details>
  
## 🧹 Brushes

  <img src="./assets/brushes_procreate.jpg" height=250>
