import os

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

FIGURE_PATH = "outputs/figures"


def create_output_directory():

    os.makedirs(
        FIGURE_PATH,
        exist_ok=True
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def plot_target_distribution(df):

    create_output_directory()

    counts = df[
        "fatal_crash"
    ].value_counts()

    plt.figure(
        figsize=(8, 6)
    )

    sns.barplot(
        x=counts.index,
        y=counts.values
    )

    plt.title(
        "Fatal vs Non-Fatal Crashes"
    )

    plt.xlabel(
        "Fatal Crash (0 = No, 1 = Yes)"
    )

    plt.ylabel(
        "Number of Crashes"
    )

    plt.tight_layout()

    plt.savefig(
        f"{FIGURE_PATH}/target_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# CRASH TYPE
# ============================================================

def plot_crash_types(df):

    create_output_directory()

    fatal_rates = (
        df.groupby(
            "first_crash_type"
        )["fatal_crash"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    plt.figure(
        figsize=(12, 7)
    )

    fatal_rates.plot(
        kind="bar"
    )

    plt.title(
        "Fatal Crash Rate by Crash Type"
    )

    plt.xlabel(
        "First Crash Type"
    )

    plt.ylabel(
        "Fatal Crash Rate"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        f"{FIGURE_PATH}/crash_types.png",
        dpi=300
    )

    plt.close()


# ============================================================
# WEATHER
# ============================================================

def plot_weather_fatality(df):

    create_output_directory()

    fatal_rates = (
        df.groupby(
            "weather_condition"
        )["fatal_crash"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    plt.figure(
        figsize=(12, 7)
    )

    fatal_rates.plot(
        kind="bar"
    )

    plt.title(
        "Fatal Crash Rate by Weather Condition"
    )

    plt.xlabel(
        "Weather Condition"
    )

    plt.ylabel(
        "Fatal Crash Rate"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        f"{FIGURE_PATH}/fatality_by_weather.png",
        dpi=300
    )

    plt.close()


# ============================================================
# LIGHTING
# ============================================================

def plot_lighting_fatality(df):

    create_output_directory()

    fatal_rates = (
        df.groupby(
            "lighting_condition"
        )["fatal_crash"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    plt.figure(
        figsize=(12, 7)
    )

    fatal_rates.plot(
        kind="bar"
    )

    plt.title(
        "Fatal Crash Rate by Lighting Condition"
    )

    plt.xlabel(
        "Lighting Condition"
    )

    plt.ylabel(
        "Fatal Crash Rate"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        f"{FIGURE_PATH}/fatality_by_lighting.png",
        dpi=300
    )

    plt.close()


# ============================================================
# CORRELATION HEATMAP
# ============================================================

def plot_correlation_heatmap(df):

    create_output_directory()

    numerical_df = df.select_dtypes(
        include=["int64", "float64"]
    )

    correlation = numerical_df.corr()

    plt.figure(
        figsize=(12, 10)
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f"
    )

    plt.title(
        "Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        f"{FIGURE_PATH}/correlation_heatmap.png",
        dpi=300
    )

    plt.close()