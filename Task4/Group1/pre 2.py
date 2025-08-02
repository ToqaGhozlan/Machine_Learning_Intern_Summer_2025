import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("final_internship_data.csv")

# تنظيف مبدئي مشابه لك (حذف القيم المفقودة، القيم السالبة، والمسافة صفر)
df.dropna(inplace=True)
df = df[df['fare_amount'] >= 0]
df = df[df['distance'] > 0]
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')

# إضافة أعمدة الوقت
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
df['pickup_month'] = df['pickup_datetime'].dt.month

# ترميز الأعمدة النوعية المهمة
important_categoricals = ['Car Condition', 'Weather', 'Traffic Condition']
df = pd.get_dummies(df, columns=important_categoricals, drop_first=True)

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10,6)

# === 1. هل هناك علاقة بين المسافة (distance) والأجرة (fare_amount)؟ ===
print("سؤال 1: هل الأجرة تزيد مع المسافة؟")
sns.regplot(data=df, x='distance', y='fare_amount', scatter_kws={'alpha':0.1}, line_kws={'color':'red'})
plt.title('Fare Amount vs Distance')
plt.xlabel('Distance (km)')
plt.ylabel('Fare Amount ($)')
plt.show()

# === 2. كيف يتأثر fare_amount بعدد الركاب؟ ===
print("سؤال 2: كيف يتغير متوسط الأجرة حسب عدد الركاب؟")
passenger_stats = df.groupby('passenger_count')['fare_amount'].agg(['mean', 'count'])
print(passenger_stats)

sns.barplot(data=df, x='passenger_count', y='fare_amount', estimator=np.mean)
plt.title('Average Fare by Passenger Count')
plt.xlabel('Passenger Count')
plt.ylabel('Average Fare ($)')
plt.show()

# === 3. هل تختلف الأجرة حسب ساعة الالتقاط؟ ===
print("سؤال 3: كيف يتغير متوسط الأجرة حسب ساعة اليوم؟")
hourly_avg = df.groupby('pickup_hour')['fare_amount'].mean()
print(hourly_avg)

sns.lineplot(x=hourly_avg.index, y=hourly_avg.values, marker='o')
plt.title('Average Fare by Pickup Hour')
plt.xlabel('Pickup Hour')
plt.ylabel('Average Fare ($)')
plt.xticks(range(0,24))
plt.show()

# === 4. هل تؤثر أيام الأسبوع على الأجرة؟ ===
print("سؤال 4: كيف تختلف الأجرة حسب يوم الأسبوع؟")
weekday_avg = df.groupby('pickup_weekday')['fare_amount'].mean()
print(weekday_avg)

sns.lineplot(x=weekday_avg.index, y=weekday_avg.values, marker='o')
plt.title('Average Fare by Weekday')
plt.xlabel('Weekday (0=Monday)')
plt.ylabel('Average Fare ($)')
plt.show()

# === 5. هل يؤثر الطقس على الأجرة؟ ===
print("سؤال 5: كيف يؤثر الطقس على متوسط الأجرة؟")
weather_cols = [col for col in df.columns if 'Weather_' in col]
weather_avg = {col.replace('Weather_', ''): df[df[col]==1]['fare_amount'].mean() for col in weather_cols}
weather_df = pd.DataFrame(list(weather_avg.items()), columns=['Weather', 'Avg Fare'])
print(weather_df)

sns.barplot(x='Weather', y='Avg Fare', data=weather_df)
plt.title('Average Fare by Weather Condition')
plt.xlabel('Weather')
plt.ylabel('Average Fare ($)')
plt.show()

# === 6. هل تؤثر حالة السيارة على الأجرة؟ ===
print("سؤال 6: تأثير حالة السيارة على الأجرة:")
car_cols = [col for col in df.columns if 'Car Condition_' in col]
car_avg = {col.replace('Car Condition_', ''): df[df[col]==1]['fare_amount'].mean() for col in car_cols}
car_df = pd.DataFrame(list(car_avg.items()), columns=['Car Condition', 'Avg Fare'])
print(car_df)

sns.barplot(x='Car Condition', y='Avg Fare', data=car_df)
plt.title('Average Fare by Car Condition')
plt.xlabel('Car Condition')
plt.ylabel('Average Fare ($)')
plt.show()

# === 7. كيف يؤثر حالة المرور على الأجرة؟ ===
print("سؤال 7: تأثير حالة المرور على الأجرة:")
traffic_cols = [col for col in df.columns if 'Traffic Condition_' in col]
traffic_avg = {col.replace('Traffic Condition_', ''): df[df[col]==1]['fare_amount'].mean() for col in traffic_cols}
traffic_df = pd.DataFrame(list(traffic_avg.items()), columns=['Traffic Condition', 'Avg Fare'])
print(traffic_df)

sns.barplot(x='Traffic Condition', y='Avg Fare', data=traffic_df)
plt.title('Average Fare by Traffic Condition')
plt.xlabel('Traffic Condition')
plt.ylabel('Average Fare ($)')
plt.show()
# === 8.1 فحص Outliers في المسافات القصيرة وأجرة مرتفعة ===
print("سؤال 8.1: هل هناك Outliers في الرحلات القصيرة؟ (مثلاً مسافة < 2 كم وأجرة > 100$)")

short_trips = df[df['distance'] < 2]
short_outliers = short_trips[short_trips['fare_amount'] > 100]

print(f"عدد الرحلات القصيرة ذات الأجرة > 100$: {len(short_outliers)}")
if not short_outliers.empty:
    print("عرض بعض الأمثلة للرحلات الشاذة:")
    print(short_outliers[['distance', 'fare_amount', 'pickup_hour', 'passenger_count']].sort_values(by='fare_amount', ascending=False).head())

# رسم Boxplot لرحلات المسافات القصيرة
sns.boxplot(x=short_trips['fare_amount'])
plt.title('Fare Amount Boxplot - Short Trips (< 2km)')
plt.xlabel('Fare Amount ($)')
plt.show()
# === 8.1 تحليل إضافي للقيم الشاذة: مسافة قصيرة + أجرة عالية ===
print("تحليل إضافي للرحلات القصيرة ذات الأجرة العالية (distance < 2km و fare_amount > 100):")
outlier_df = df[(df['distance'] < 2) & (df['fare_amount'] > 100)]

# 1. عددها
print(f"عدد الرحلات الشاذة: {outlier_df.shape[0]}")

# 2. إحصائيات وصفية
print(outlier_df[['fare_amount', 'distance', 'pickup_hour', 'passenger_count']].describe())

# 3. توزيع حسب ساعة الالتقاط
plt.figure(figsize=(8,4))
sns.countplot(data=outlier_df, x='pickup_hour')
plt.title('Distribution of anomalous flights by pickup time')
plt.xlabel('Hour')
plt.ylabel('Count')
plt.show()

# 4. توزيع حسب عدد الركاب
plt.figure(figsize=(6,4))
sns.countplot(data=outlier_df, x='passenger_count')
plt.title('Number of passengers on irregular flights')
plt.xlabel('Passenger Count')
plt.ylabel('Count')
plt.show()

# 5. توزيع حسب يوم الأسبوع
plt.figure(figsize=(6,4))
sns.countplot(data=outlier_df, x='pickup_weekday')
plt.title('Distribution of irregular flights by day of the week')
plt.xlabel('Weekday (0=Monday)')
plt.ylabel('Count')
plt.show()

# 6. أجرة مقابل المسافة في الرحلات الشاذة
plt.figure(figsize=(8,6))
sns.scatterplot(data=outlier_df, x='distance', y='fare_amount', hue='pickup_hour', palette='coolwarm')
plt.title('Fare vs Distance (For irregular trips only)')
plt.xlabel('Distance (km)')
plt.ylabel('Fare ($)')
plt.show()
# حذف القيم الشاذة: رحلات قصيرة جدًا مع أجرة عالية بشكل غير منطقي
outliers_condition = (df['distance'] < 2) & (df['fare_amount'] > 100)
num_outliers = outliers_condition.sum()

df = df[~outliers_condition]

print(f"🧹 تم حذف {num_outliers} رحلة شاذة (مسافة < 2 كم وأجرة > 100$) من البيانات.")
