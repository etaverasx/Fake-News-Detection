import pandas as pd

# Load the Excel file
file_path = 'news_articles-1.xlsx'
df = pd.read_excel(file_path)

print("✅ Data loaded successfully!")
print("Shape:", df.shape)
print(df.head())
