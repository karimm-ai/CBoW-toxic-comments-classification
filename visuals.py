import matplotlib.pyplot as plt
import seaborn as sns


def count_classes(df, labels):
    label_counts = df[labels].sum().sort_values(ascending=False)
    print(label_counts)

    plt.figure(figsize=(10,5))
    sns.barplot(
        x=label_counts.index,
        y=label_counts.values,
        palette="viridis"
    )

    plt.title("Label Distribution")
    plt.ylabel("Number of Comments")
    plt.xlabel("Class")
    plt.show()


def classes_percentage(df, labels):
    label_counts = df[labels].sum().sort_values(ascending=False)
    percentages = label_counts / len(df) * 100

    plt.figure(figsize=(10,5))
    sns.barplot(
        x=percentages.index,
        y=percentages.values,
        palette="magma"
    )

    plt.ylabel("Percentage (%)")
    plt.title("Class Percentages")
    plt.show()


def comment_labels_count(df, labels):
    df['num_labels'] = df[labels].sum(axis=1)

    sns.countplot(
        x='num_labels',
        data=df,
        color='steelblue'
    )

    plt.title("Number of Labels per Comment")
    plt.xlabel("Labels Assigned")
    plt.ylabel("Count")
    plt.show()


def labels_corr(df, labels):
    plt.figure(figsize=(8,6))

    sns.heatmap(
        df[labels].corr(),
        annot=True,
        cmap='coolwarm'
    )

    plt.title("Label Correlation Matrix")
    plt.show()


def comment_len(df, comment_col):
    df['char_length'] = df[comment_col].str.len()

    plt.figure(figsize=(10,5))

    sns.histplot(
        df['char_length'],
        bins=50,
        kde=True
    )

    plt.title("Comment Length Distribution")
    plt.show()