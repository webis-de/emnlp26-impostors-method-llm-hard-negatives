from datasets import load_from_disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from genai_detection.config import CONFIG

import genai_detection.detectors.llm_unmasking as unm

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "Helvetica, sans-serif"
plt.rcParams["font.size"] = 8
plt.rcParams["lines.linewidth"] = 1
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["grid.linewidth"] = 0.25


def split_pan_ds_dataset_by_model_pan(df, seed=42):
    return {
        "Human": df.query('model == "human"').sample(frac=1, random_state=seed),
        # 'GPT3': df.query('model.str.startswith("gpt-3")').sample(frac=1, random_state=seed),
        # 'GPT4': df.query('model.str.startswith("gpt-4")').sample(frac=1, random_state=seed),
        # 'o1': df.query('model.str.startswith("openai")').sample(frac=1, random_state=seed),
        # 'Llama2': df.query('model.str.startswith("llama")').sample(frac=1, random_state=seed),
        # 'PaLM2': df.query('model.str.startswith("text-bison")').sample(frac=1, random_state=seed),
        # 'Gemini': df.query('model.str.startswith("gemini")').sample(frac=1, random_state=seed),
        "gpt-4-turbo-paraphrase": df.query('model == "gpt-4-turbo-paraphrase"').sample(
            frac=1, random_state=seed
        ),
        "gemini-pro": df.query('model == "gemini-pro"').sample(
            frac=1, random_state=seed
        ),
        "gpt-4-turbo": df.query('model == "gpt-4-turbo"').sample(
            frac=1, random_state=seed
        ),
        "gemini-pro-paraphrase": df.query('model == "gemini-pro-paraphrase"').sample(
            frac=1, random_state=seed
        ),
        # 'Mistral': df.query('model.str.startswith("mistral") or model.str.startswith("mixtral")').sample(
        # frac=1, random_state=seed),
        # 'Qwen': df.query('model.str.startswith("qwen")').sample(frac=1, random_state=seed),
    }


def split_pan_ds_dataset_by_model_hd(df, seed=42):
    return {
        "Human": df.query('model == "human"').sample(frac=1, random_state=seed),
        # 'Claude': df.query('model.str.startswith("claude")').sample(frac=1, random_state=seed),
        "GPT-4o": df.query('model.str.startswith("gpt-4o")').sample(
            frac=1, random_state=seed
        ),
        # 'o1': df.query('model.str.startswith("o1-pro")').sample(frac=1, random_state=seed),
        # 'o1 (Humanized)': df.query('model.str.startswith("humanized_o1")').sample(frac=1, random_state=seed),
        "GPT-4o (Paraphrased)": df.query(
            'model.str.startswith("paraphrased_gpt")'
        ).sample(frac=1, random_state=seed),
    }


ds = load_from_disk(CONFIG.PATH2PAN25)["train"].to_pandas()
ds_by_model = split_pan_ds_dataset_by_model_pan(ds)
# ds = load_from_disk('data/datasets/human-detectors-converted')['train'].to_pandas()
# ds_by_model = split_pan_ds_dataset_by_model_hd(ds)


def unmask(labels, texts, **unmasking_kwargs):
    unmask_det = unm.LLMUnmasking(**unmasking_kwargs)

    curves = []
    stripplot = False
    for l, t in zip(labels, texts):
        c = unmask_det.get_curves(t, 6)
        # c = np.diff(unmask_det.get_curves(t, 6))
        if c.shape[-1] == 1:
            stripplot = True
            c = np.sum(c, axis=-1, keepdims=True)
        c = pd.DataFrame({f"{i:04}_{l}": c for i, c in enumerate(c)})
        c["model"] = l
        curves.append(c)
    curves = pd.concat(curves)
    curves = pd.melt(
        curves,
        var_name="curve_id",
        value_name="nll",
        id_vars=["model"],
        ignore_index=False,
    )
    curves = curves.reset_index(drop=False, names="round")

    plt.figure(figsize=(4, 4))
    if stripplot:
        ax = sns.stripplot(curves, x="model", y="nll", hue="model")
        sns.boxplot(
            curves,
            ax=ax,
            x="model",
            y="nll",
            zorder=10,
            showfliers=False,
            showbox=True,
            showcaps=True,
            boxprops={"facecolor": "none"},
        )
        ax.set_xlabel("Model")
        # ax.set_ylim((-1.2, -0.1))
    else:
        ax = sns.lineplot(
            curves,
            x="round",
            y="nll",
            hue="model",
            style="model",
            units="curve_id",
            estimator=None,
        )
        ax.set_xlabel("Rounds")
        ax.legend(title="Model")
    ax.set_ylabel("Inverse negative log likelihood")
    # ax.set_yscale('log')
    plt.tight_layout()
    plt.show()


min_text_ln = 3000
unmask(
    ds_by_model.keys(),
    [
        m.query("text.str.len() >= @min_text_ln")["text"][:20]
        for m in ds_by_model.values()
    ],
    base_model="tiiuae/Falcon3-7B-Instruct",
    quantization_bits=4,
    device="cuda",
)
