{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyNK9llx4aubmcMJRDDut8CG",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Ojasvi310/PixelForge/blob/main/backend/src/train_text_adapter.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "r9ctSzZtc9IH"
      },
      "outputs": [],
      "source": [
        "# src/train_text_adapter.py\n",
        "import os\n",
        "from datasets import load_dataset\n",
        "from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments\n",
        "from peft import LoraConfig, get_peft_model, TaskType\n",
        "\n",
        "MODEL = \"bert-base-multilingual-cased\"\n",
        "DATA_CSV = \"/content/project/data/text_corpus.csv\"  # can be single-column CSV with 'text' column\n",
        "\n",
        "def main():\n",
        "    tokenizer = AutoTokenizer.from_pretrained(MODEL)\n",
        "    model = AutoModelForMaskedLM.from_pretrained(MODEL)\n",
        "\n",
        "    # Add LoRA adapters\n",
        "    lora_config = LoraConfig(\n",
        "        task_type=TaskType.MLM,\n",
        "        r=8,\n",
        "        lora_alpha=32,\n",
        "        lora_dropout=0.1,\n",
        "    )\n",
        "    model = get_peft_model(model, lora_config)\n",
        "\n",
        "    dataset = load_dataset(\"csv\", data_files={\"train\": DATA_CSV})\n",
        "    def tokenize_fn(ex):\n",
        "        return tokenizer(ex[\"text\"], truncation=True, padding=\"max_length\", max_length=256)\n",
        "    tokenized = dataset.map(tokenize_fn, batched=True)\n",
        "    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)\n",
        "\n",
        "    args = TrainingArguments(\n",
        "        output_dir=\"/content/project/models/text_adapter\",\n",
        "        per_device_train_batch_size=8,\n",
        "        num_train_epochs=3,\n",
        "        logging_steps=50,\n",
        "        fp16=True,\n",
        "        save_total_limit=2,\n",
        "        learning_rate=5e-5,\n",
        "    )\n",
        "    trainer = Trainer(model=model, args=args, train_dataset=tokenized[\"train\"], data_collator=data_collator)\n",
        "    trainer.train()\n",
        "    model.save_pretrained(args.output_dir)\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    main()\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "F9sO40x1dLGu"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}