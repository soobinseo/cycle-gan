# Cycle-GAN
tensorflow implementation of [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) (Cycle-GAN)

I used horse2zebra dataset, and I did not use all number of residual blocks that the paper demonstrated (9 --> 6).
The structure of model is below.

<p>
  <img src="https://raw.githubusercontent.com/soobin3230/cycle-gan/master/png/cyclegan.png" width="1024"/>
</p>

## Dependencies

1. tensorflow >= 1.0.0
1. numpy == 1.12.0
1. matplotlib == 1.3.1

## Steps

Run the following code for image generation.

<pre><code>
python model.py
</code></pre>
- There is no evalution code, and I will update soon.

## Results

- The model trained 1000 epochs

- Generate Horse to Zebra Image
<p>
  <img src="https://raw.githubusercontent.com/soobin3230/cycle-gan/master/png/AB_1.png" width="112"/>
  <img src="https://raw.githubusercontent.com/soobin3230/cycle-gan/master/png/AB_2.png" width="112"/>
</p>

- Generate Zebra to Horse Image
<p>
  <img src="https://raw.githubusercontent.com/soobin3230/ARAE/master/png/real_42000.png" width="112"/>
  <img src="https://raw.githubusercontent.com/soobin3230/ARAE/master/png/real_42000.png" width="112"/>
  <img src="https://raw.githubusercontent.com/soobin3230/ARAE/master/png/real_42000.png" width="112"/>
</p>

## Notes

I didn't multiply the critic gradient before backpropping to the encoder.