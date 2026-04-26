from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import nbformat as nbf


ROOT = Path(__file__).resolve().parent


def _write_notebook(path: Path, cells: Sequence[nbf.NotebookNode]) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = list(cells)
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }
    nbf.write(nb, path)
    print(f"[build] wrote {path.name}")


def build_market() -> None:
    path = ROOT / "Model market.ipynb"

    intro = nbf.v4.new_markdown_cell(
        "## Прогноз ринку (three-phase linear)\n\n"
        "Зошит повторює ключові кроки `3p_linear_model`: базовий Holt-Winters, "
        "сезонні та лагові ознаки, фінальна модель XGBoost для кожної товарної групи."
    )

    imports = nbf.v4.new_code_cell(
        "from pathlib import Path\n\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "from three_phase_linear import ForecastConfig, run_three_phase_forecast\n\n"
        "DATA_PATH = Path('forecast_of_market_dataset.csv')\n"
        "OUTPUT_PATH = Path('market_three_phase_forecast.csv')\n"
        "GROUP_COLS = ['product_group_id']\n"
        "TARGET_COLUMNS = ['market_revenue', 'revenue_amazon']"
    )

    prepare = nbf.v4.new_code_cell(
        "df = pd.read_csv(DATA_PATH)\n"
        "df['month'] = pd.to_datetime(df['month'])\n"
        "df = df.sort_values(GROUP_COLS + ['month']).reset_index(drop=True)\n\n"
        "future_counts = df[df['market_revenue'].isna()].groupby(GROUP_COLS).size()\n"
        "forecast_horizon = int(future_counts.max()) if not future_counts.empty else 12\n"
        "if forecast_horizon <= 0:\n"
        "    forecast_horizon = 12\n\n"
        "print(f'Горизонт прогнозу: {forecast_horizon} періодів')"
    )

    run_pipeline = nbf.v4.new_code_cell(
        "prediction_frames = {}\n"
        "summary_frames = []\n\n"
        "for target in TARGET_COLUMNS:\n"
        "    target_df = df[['month', *GROUP_COLS, target]].copy()\n"
        "    config = ForecastConfig(\n"
        "        time_col='month',\n"
        "        target_col=target,\n"
        "        group_cols=GROUP_COLS,\n"
        "        freq='MS',\n"
        "        forecast_horizon=forecast_horizon,\n"
        "        seasonal_periods=12,\n"
        "        min_history=24,\n"
        "        lags=(1, 2, 3, 6, 12, 18, 24),\n"
        "        rolling_windows=(3, 6, 12, 24),\n"
        "        random_search_iterations=10,\n"
        "        n_splits=4,\n"
        "        random_state=46,\n"
        "    )\n\n"
        "    preds, summaries = run_three_phase_forecast(target_df, config)\n"
        "    preds = preds.rename(columns={\n"
        "        'prediction': f'{target}_forecast',\n"
        "        f'{target}_holtwinters': f'{target}_baseline',\n"
        "    })\n"
        "    prediction_frames[target] = preds\n\n"
        "    summary_df = pd.DataFrame({\n"
        "        'group_key': [s.group_key[0] for s in summaries],\n"
        "        'train_rows': [s.train_rows for s in summaries],\n"
        "        'cv_mae': [s.best_score for s in summaries],\n"
        "        'skipped_reason': [s.skipped_reason for s in summaries],\n"
        "    })\n"
        "    summary_df['target'] = target\n"
        "    summary_frames.append(summary_df)\n\n"
        "summary_report = pd.concat(summary_frames, ignore_index=True)\n"
        "summary_report.head()"
    )

    merge_save = nbf.v4.new_code_cell(
        "result_df = df.copy()\n"
        "original_masks = {target: result_df[target].isna() for target in TARGET_COLUMNS}\n\n"
        "for target, preds in prediction_frames.items():\n"
        "    merge_cols = [*GROUP_COLS, 'month']\n"
        "    result_df = result_df.merge(\n"
        "        preds[merge_cols + [f'{target}_forecast']],\n"
        "        on=merge_cols,\n"
        "        how='left'\n"
        "    )\n"
        "    result_df[target] = result_df[target].astype(float)\n"
        "    result_df[target] = result_df[target].fillna(result_df[f'{target}_forecast'])\n\n"
        "output_columns = ['month', 'product_group_id', 'product_group_name', 'market_revenue', 'revenue_amazon']\n"
        "forecast_mask = np.zeros(len(result_df), dtype=bool)\n"
        "for target, mask in original_masks.items():\n"
        "    forecast_mask |= mask\n"
        "final_output = result_df.loc[forecast_mask, output_columns].sort_values(['product_group_id', 'month']).reset_index(drop=True)\n"
        "final_output.to_csv(OUTPUT_PATH, index=False)\n\n"
        "final_output.tail()"
    )

    report = nbf.v4.new_code_cell("summary_report")

    _write_notebook(path, [intro, imports, prepare, run_pipeline, merge_save, report])


def build_money() -> None:
    path = ROOT / "Model money.ipynb"

    intro = nbf.v4.new_markdown_cell(
        "## Прогноз виручки (three-phase linear)\n\n"
        "Щомісячний прогноз виручки по категоріях із повторним використанням пайплайну."
    )

    imports = nbf.v4.new_code_cell(
        "from pathlib import Path\n\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "from three_phase_linear import ForecastConfig, run_three_phase_forecast\n\n"
        "DATA_PATH = Path('forecast_revenue_dataset.csv')\n"
        "OUTPUT_PATH = Path('money_three_phase_forecast.csv')\n"
        "GROUP_COLS = ['category_id']\n"
        "TARGET_COLUMN = 'revenue'\n"
        "REGRESSORS = ['is_sale_prohibition', 'cos_month', 'sin_month', 'cos_quarter', 'sin_quarter', 'unique_brand_count']"
    )

    prepare = nbf.v4.new_code_cell(
        "df = pd.read_csv(DATA_PATH, sep=';')\n"
        "df['date'] = pd.to_datetime(df['date'], dayfirst=True)\n"
        "df = df.sort_values(GROUP_COLS + ['date']).reset_index(drop=True)\n"
        "for col in REGRESSORS:\n"
        "    df[col] = pd.to_numeric(df[col], errors='coerce')\n"
        "\n"
        "agg_df = df.groupby(['date', 'category_id', 'category_title'], as_index=False).agg(\n"
        "    revenue_sum=('revenue', 'sum'),\n"
        "    revenue_count=('revenue', 'count'),\n"
        "    is_sale_prohibition=('is_sale_prohibition', 'max'),\n"
        "    cos_month=('cos_month', 'mean'),\n"
        "    sin_month=('sin_month', 'mean'),\n"
        "    cos_quarter=('cos_quarter', 'mean'),\n"
        "    sin_quarter=('sin_quarter', 'mean'),\n"
        "    unique_brand_count=('unique_brand_count', 'mean'),\n"
        ")\n"
        "agg_df.loc[agg_df['revenue_count'] == 0, 'revenue_sum'] = np.nan\n"
        "df = agg_df.rename(columns={'revenue_sum': 'revenue'}).drop(columns=['revenue_count'])\n"
        "df = df.sort_values(GROUP_COLS + ['date']).reset_index(drop=True)\n\n"
        "future_counts = df[df[TARGET_COLUMN].isna()].groupby(GROUP_COLS).size()\n"
        "forecast_horizon = int(future_counts.max()) if not future_counts.empty else 12\n"
        "if forecast_horizon <= 0:\n"
        "    forecast_horizon = 12\n\n"
        "print(f'Горизонт прогнозу: {forecast_horizon} періодів')"
    )

    run_pipeline = nbf.v4.new_code_cell(
        "input_cols = ['date', *GROUP_COLS, TARGET_COLUMN, *REGRESSORS]\n"
        "input_cols = list(dict.fromkeys(input_cols))\n"
        "config = ForecastConfig(\n"
        "    time_col='date',\n"
        "    target_col=TARGET_COLUMN,\n"
        "    group_cols=GROUP_COLS,\n"
        "    freq='MS',\n"
        "    forecast_horizon=forecast_horizon,\n"
        "    seasonal_periods=12,\n"
        "    min_history=24,\n"
        "    lags=(1, 2, 3, 6, 12, 18, 24),\n"
        "    rolling_windows=(3, 6, 12, 24),\n"
        "    additional_regressors=REGRESSORS,\n"
        "    random_search_iterations=10,\n"
        "    n_splits=4,\n"
        "    random_state=46,\n"
        ")\n\n"
        "preds, summaries = run_three_phase_forecast(df[input_cols].copy(), config)\n"
        "preds = preds.rename(columns={\n"
        "    'prediction': 'revenue_forecast',\n"
        "    f'{TARGET_COLUMN}_holtwinters': 'revenue_baseline',\n"
        "})\n"
        "summary_report = pd.DataFrame({\n"
        "    'group_key': [s.group_key[0] for s in summaries],\n"
        "    'train_rows': [s.train_rows for s in summaries],\n"
        "    'cv_mae': [s.best_score for s in summaries],\n"
        "    'skipped_reason': [s.skipped_reason for s in summaries],\n"
        "})\n"
        "summary_report.head()"
    )

    merge_save = nbf.v4.new_code_cell(
        "merge_cols = [*GROUP_COLS, 'date']\n"
        "result_df = df.copy()\n"
        "result_df['is_forecast_period'] = result_df[TARGET_COLUMN].isna()\n"
        "result_df = result_df.merge(preds[merge_cols + ['revenue_forecast']], on=merge_cols, how='left')\n"
        "result_df['revenue'] = result_df['revenue'].astype(float)\n"
        "result_df['revenue'] = result_df['revenue'].fillna(result_df['revenue_forecast'])\n\n"
        "output_columns = ['date', 'category_id', 'category_title', 'revenue']\n"
        "final_output = result_df.loc[result_df['is_forecast_period'], output_columns]\n"
        "final_output = final_output.groupby(['date', 'category_id', 'category_title'], as_index=False)['revenue'].sum()\n"
        "final_output = final_output.sort_values(['category_id', 'date']).reset_index(drop=True)\n"
        "final_output.to_csv(OUTPUT_PATH, index=False)\n\n"
        "final_output.tail()"
    )

    report = nbf.v4.new_code_cell("summary_report")

    _write_notebook(path, [intro, imports, prepare, run_pipeline, merge_save, report])


def build_pcs() -> None:
    path = ROOT / "Model pcs.ipynb"

    intro = nbf.v4.new_markdown_cell(
        "## Прогноз PCS (three-phase linear)\n\n"
        "Прогноз щотижневих продажів по SKU з додатковими регресорами."
    )

    imports = nbf.v4.new_code_cell(
        "from pathlib import Path\n\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "from three_phase_linear import ForecastConfig, run_three_phase_forecast\n\n"
        "DATA_PATH = Path('dataset_pcs.csv')\n"
        "OUTPUT_PATH = Path('pcs_three_phase_forecast.csv')\n"
        "GROUP_COLS = ['sku_id']\n"
        "TARGET_COLUMN = 'qty_total'\n"
        "REGRESSORS = [\n"
        "    'orders_qty', 'total_abc_numeric', 'avg_discount_perc_by_goods',\n"
        "    'max_discount_perc_by_goods', 'avg_goods_price_by_goods', 'oos__by_goods',\n"
        "    'war', 'covid', 'sin_quarter', 'cos_quarter', 'sin_month', 'cos_month',\n"
        "    'sin_week', 'cos_week'\n"
        "]"
    )

    prepare = nbf.v4.new_code_cell(
        "df = pd.read_csv(DATA_PATH)\n"
        "df['period'] = pd.to_datetime(df['period'])\n"
        "comma_cols = ['avg_discount_perc_by_goods', 'max_discount_perc_by_goods', 'avg_goods_price_by_goods', 'oos__by_goods', 'sin_month', 'cos_month', 'sin_week', 'cos_week']\n"
        "for col in comma_cols:\n"
        "    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)\n"
        "    df[col] = pd.to_numeric(df[col], errors='coerce')\n"
        "numeric_cols = ['qty_total', 'orders_qty', 'total_abc_numeric', 'war', 'covid', 'sin_quarter', 'cos_quarter']\n"
        "for col in numeric_cols:\n"
        "    df[col] = pd.to_numeric(df[col], errors='coerce')\n"
        "\n"
        "# Mark placeholder rows (future horizon) where target should be forecasted\n"
        "placeholder_mask = df['last_goods_sell_status'].isna() & df['oos__by_goods'].isna()\n"
        "df.loc[placeholder_mask, TARGET_COLUMN] = np.nan\n"
        "\n"
        "df = df.sort_values(GROUP_COLS + ['period']).reset_index(drop=True)\n\n"
        "forecast_horizon = int(df.loc[placeholder_mask, 'period'].nunique()) if placeholder_mask.any() else 0\n"
        "if forecast_horizon <= 0:\n"
        "    forecast_horizon = 4\n"
        "print(f'Горизонт прогнозу: {forecast_horizon} тижнів')"
    )

    run_pipeline = nbf.v4.new_code_cell(
        "input_cols = ['period', *GROUP_COLS, 'category_id', TARGET_COLUMN, *REGRESSORS]\n"
        "input_cols = list(dict.fromkeys(input_cols))\n"
        "config = ForecastConfig(\n"
        "    time_col='period',\n"
        "    target_col=TARGET_COLUMN,\n"
        "    group_cols=GROUP_COLS,\n"
        "    freq='W-MON',\n"
        "    forecast_horizon=forecast_horizon,\n"
        "    seasonal_periods=52,\n"
        "    min_history=20,\n"
        "    lags=(1, 2, 3, 4, 8, 12, 16),\n"
        "    rolling_windows=(3, 4, 8, 12),\n"
        "    additional_regressors=REGRESSORS,\n"
        "    random_search_iterations=0,\n"
        "    n_splits=3,\n"
        "    n_estimators=300,\n"
        "    target_transform=np.log1p,\n"
        "    target_inverse_transform=np.expm1,\n"
        "    random_state=46,\n"
        ")\n\n"
        "preds, summaries = run_three_phase_forecast(df[input_cols].copy(), config)\n"
        "preds = preds.rename(columns={\n"
        "    'prediction': 'qty_total_forecast',\n"
        "    f'{TARGET_COLUMN}_holtwinters': 'qty_total_baseline',\n"
        "})\n"
        "preds['qty_total_forecast'] = preds['qty_total_baseline']\n"
        "summary_report = pd.DataFrame({\n"
        "    'group_key': [s.group_key[0] for s in summaries],\n"
        "    'train_rows': [s.train_rows for s in summaries],\n"
        "    'cv_mae': [s.best_score for s in summaries],\n"
        "    'skipped_reason': [s.skipped_reason for s in summaries],\n"
        "})\n"
        "summary_report.head()"
    )

    merge_save = nbf.v4.new_code_cell(
        "merge_cols = [*GROUP_COLS, 'period']\n"
        "forecast_df = preds[merge_cols + ['qty_total_forecast']].copy()\n"
        "static_map = df[['sku_id', 'category_id']].drop_duplicates()\n"
        "forecast_df = forecast_df.merge(static_map, on='sku_id', how='left')\n"
        "forecast_df = forecast_df.rename(columns={'qty_total_forecast': 'qty_total'})\n"
        "forecast_df = forecast_df.sort_values(GROUP_COLS + ['period']).reset_index(drop=True)\n"
        "forecast_df.to_csv(OUTPUT_PATH, index=False)\n\n"
        "forecast_df.tail()"
    )

    report = nbf.v4.new_code_cell("summary_report")

    _write_notebook(path, [intro, imports, prepare, run_pipeline, merge_save, report])


def main() -> None:
    build_market()
    build_money()
    build_pcs()


if __name__ == "__main__":
    main()
