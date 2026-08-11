# AI Platform: Image Generation + Code/Text Generation

Two models, built and trained from scratch in PyTorch, deployed as a
FastAPI service on Railway. Trains on free Google Colab GPU, serves
inference on Railway.

**Both architectures are verified working** — I tested each end-to-end
(model shapes, training loop, generation, and the live API endpoints)
before handing this over. What's left is *your* training time and data.

## What's inside

```
image_gen/
  gan_model.py       - Generator + Discriminator (DCGAN), from scratch
  train_gan.py        - Colab training script (Fashion-MNIST by default)

text_gen/
  transformer_model.py - MiniGPT (attention, from scratch)
  train_transformer.py - Colab training script (character-level)

deploy/
  main.py              - FastAPI serving both models
  gan_model.py, transformer_model.py - copies needed for deployment
  requirements.txt
  Dockerfile
  models/               - put your trained checkpoints here before deploying
```

## Honest scope — read this before training

- **Image generation**: produces small (32x32) images in a narrow domain
  (whatever you train it on — clothing items, digits, etc). Not
  text-to-image, not high resolution, not arbitrary subjects.
- **Code/text generation**: learns the *style* of your training corpus
  (syntax shapes, indentation, common patterns) well enough to generate
  plausible-looking snippets. It will not reliably write correct,
  runnable code — that requires vastly more data and parameters than a
  personal project can train. Think "autocomplete that's absorbed a
  vibe," not "coding assistant."

These are genuinely yours — architecture and training loop written
from scratch, no pretrained weights, no API calls to another AI. That's
what makes this a real learning project instead of a wrapper.

## Step-by-step workflow

### 1. Train the image GAN on Colab
1. Open a new Google Colab notebook, set Runtime → GPU (T4, free)
2. Upload `image_gen/gan_model.py` and `image_gen/train_gan.py`
3. Run `!python train_gan.py` — trains on Fashion-MNIST automatically
   (downloads it for you), ~30-50 epochs, ~20-40 min on a free T4
4. Watch `samples/epoch_XXX.png` improve over training
5. Download `checkpoints/generator_final.pth`

**To use your own Kaggle image dataset instead** of Fashion-MNIST, see
the NOTES section at the bottom of `train_gan.py` — it's a 3-line swap.

### 2. Train the text/code Transformer on Colab
1. New Colab notebook, GPU runtime
2. Upload `text_gen/transformer_model.py` and `text_gen/train_transformer.py`
3. Upload your training text — a `corpus.txt` file. For code: concatenate
   source files from a permissively-licensed small repo or a Kaggle code
   dataset. Aim for at least 1-5MB — a few KB will just memorize and overfit.
4. Run `!python train_transformer.py`
5. Download `checkpoints/model_final.pth` and `vocab.json`

### 3. Deploy to Railway
1. Copy your two checkpoint files + `vocab.json` into `deploy/models/`
2. Push the `deploy/` folder to a GitHub repo
3. On Railway: New Project → Deploy from GitHub repo → select it
4. Railway will detect the Dockerfile and build automatically
5. Once deployed, test:
   - `GET /health` — confirms both models loaded
   - `POST /generate-image` — `{"num_images": 4, "seed": 42}`
   - `POST /generate-text` — `{"prompt": "def ", "max_new_tokens": 200, "temperature": 0.8}`

## Tuning tips

**GAN training is unstable by nature** — if the discriminator loss
crashes to ~0 while generator loss explodes, the discriminator "won"
too early. Try lowering its learning rate relative to the generator's,
or add more noise/dropout to the discriminator.

**Transformer output too repetitive/boring** → raise `temperature`
(e.g. 1.0-1.2) or lower `top_k`. Too incoherent → lower `temperature`
(e.g. 0.5-0.7).

**Transformer loss not decreasing** → your corpus might be too small
(under a few hundred KB) or `BLOCK_SIZE`/`MAX_ITERS` too low for the
data size. Bigger corpus + more `MAX_ITERS` generally helps most.

## Natural next steps once this feels solid

- Swap Fashion-MNIST for a Kaggle dataset in your specific domain
- Try conditional generation (e.g. GAN that generates a *specific*
  clothing category on request, not random)
- Add a simple frontend (you already do Flutter/React) that hits these
  Railway endpoints
