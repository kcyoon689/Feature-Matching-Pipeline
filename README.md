# PROJECT NAME

## Pipeline Modules

1. Feature Extractor
2. Feature Descriptor
3. Feature Matcher
4. Transformation Calculator

### Weights for GlueStick

#### 1. Extract from Provided Archive

If you use the compressed file (`resources.7z`) included in the repository, extract it with the following command:

```bash
7z x ./resources.7z
# or
7za x ./resources.7z
```

Note: You may need to install 7z if it is not already available:

```bash
sudo apt install p7zip-full
```

The extracted files will be placed in the appropriate directory (e.g., resources/weights).

#### 2. Download Directly

Alternatively, you can download the official model file from the GlueStick repository:

```bash
wget https://github.com/cvg/GlueStick/releases/download/v0.1_arxiv/checkpoint_GlueStick_MD.tar -P resources/weights
```
