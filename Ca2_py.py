import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"C:\Users\USER\Downloads\ecommerce_returns_synthetic_data.csv")


df['Order_Date'] = pd.to_datetime(df['Order_Date'], dayfirst=True, errors='coerce')
df['Return_Date'] = pd.to_datetime(df['Return_Date'], dayfirst=True, errors='coerce')


df['Days_to_Return'] = (df['Return_Date'] - df['Order_Date']).dt.days
df['Days_to_Return'] = df['Days_to_Return'].fillna(0)


df['Is_Returned'] = np.where(df['Return_Status'] == 'Returned', 1, 0)


if 'User_Age' in df.columns:
    df['Age_Group'] = pd.cut(df['User_Age'], bins=[0, 18, 25, 35, 50, 100],
                             labels=["Teen", "Young Adult", "Adult", "Mid Age", "Senior"])


base_purple_palette = ["#d8b4f8", "#b794f4", "#9f7aea", "#805ad5", "#6b46c1", "#553c9a", "#44337a"]


print("\n--- BASIC DATA INFO ---")
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)
print("\nSummary stats:\n", df.describe(include='all'))


avg_price = np.mean(df['Product_Price'])
std_price = np.std(df['Product_Price'])
median_days = np.median(df['Days_to_Return'])
unique_customers = np.unique(df['User_ID']).size

print("\n--- NUMPY INSIGHTS ---")
print("Average product price:", round(avg_price, 2))
print("Std. deviation of product price:", round(std_price, 2))
print("Median days to return:", median_days)
print("Unique customers:", unique_customers)


sns.set_theme(style="whitegrid")


plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Product_Category', y='Is_Returned', hue='Product_Category',
            estimator=np.mean, palette=base_purple_palette[:df['Product_Category'].nunique()], legend=False)
plt.title("Return Rate by Product Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Product_Category', y='Days_to_Return', hue='Product_Category',
            palette=base_purple_palette[:df['Product_Category'].nunique()], legend=False)
plt.title("Days to Return per Product Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


if 'Age_Group' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Age_Group', y='Is_Returned', hue='Age_Group',
                estimator=np.mean, palette=base_purple_palette[:df['Age_Group'].nunique()], legend=False)
    plt.title("Return Rate by Age Group")
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Purples")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


df['Price_Range'] = pd.qcut(df['Product_Price'], q=4, labels=["Low", "Medium", "High", "Premium"])
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Payment_Method', y='Product_Price', hue='Return_Status',
            palette=base_purple_palette[:df['Return_Status'].nunique()])
plt.title("Price Distribution by Payment and Return Status")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
top_returns = df[df['Return_Status'] == 'Returned']['Product_Category'].value_counts().head(10)
sns.barplot(x=top_returns.values, y=top_returns.index, hue=top_returns.index,
            palette=base_purple_palette[:len(top_returns)], legend=False)
plt.title("Top Returned Product Categories")
plt.xlabel("Number of Returns")
plt.ylabel("Product Category")
plt.tight_layout()
plt.show()


if 'Age_Group' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df[df['Is_Returned'] == 1],
                x='Age_Group',
                y='Days_to_Return',
                hue='Age_Group',
                palette=base_purple_palette[:df['Age_Group'].nunique()],
                legend=False)
    plt.title("Average Days to Return by Age Group")
    plt.tight_layout()
    plt.show()


if 'User_Gender' in df.columns:
    plt.figure(figsize=(6, 5))
    sns.countplot(data=df[df['Return_Status'] == 'Returned'],
                  x='User_Gender',
                  hue='User_Gender',
                  palette=base_purple_palette[:df['User_Gender'].nunique()],
                  legend=False)
    plt.title("Return Count by Gender")
    plt.tight_layout()
    plt.show()


df['Order_Month'] = df['Order_Date'].dt.to_period('M').astype(str)
monthly_returns = df[df['Is_Returned'] == 1].groupby('Order_Month').size()
plt.figure(figsize=(12, 5))
sns.lineplot(x=monthly_returns.index, y=monthly_returns.values, color="#7c3aed", marker="o")
plt.xticks(rotation=45)
plt.title("Monthly Return Volume Over Time")
plt.xlabel("Month")
plt.ylabel("Returns")
plt.tight_layout()
plt.show()


if 'Return_Reason' in df.columns:
    plt.figure(figsize=(7, 7))
    reason_counts = df[df['Return_Status'] == 'Returned']['Return_Reason'].value_counts()
    plt.pie(reason_counts,
            labels=reason_counts.index,
            autopct='%1.1f%%',
            colors=base_purple_palette[:len(reason_counts)],
            startangle=140)
    plt.title("Distribution of Return Reasons")
    plt.tight_layout()
    plt.show()


if 'Return_Reason' in df.columns:
    top_returns = df['Return_Reason'].value_counts().head(5)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_returns.values, y=top_returns.index, hue=top_returns.index,
                palette=base_purple_palette[:len(top_returns)], legend=False)
    plt.title("Top 5 Return Reasons")
    plt.tight_layout()
    plt.show()

# Scatter Plot: Product Price vs. Days to Return
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='Product_Price',
    y='Days_to_Return',
    hue='Return_Status',
    palette=base_purple_palette[:df['Return_Status'].nunique()],
    alpha=0.7
)
plt.title("Scatter Plot: Product Price vs. Days to Return")
plt.xlabel("Product Price")
plt.ylabel("Days to Return")
plt.tight_layout()
plt.show()

