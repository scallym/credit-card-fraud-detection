# Credit Card Fraud Detection

Мой первый проект по машинному обучению.

## Цель проекта

Цель данного проекта — построить модель машинного обучения, способную обнаруживать мошеннические транзакции по банковским картам.

## Датасет

В проекте используется датасет Credit Card Fraud Detection с Kaggle.

Информация о датасете:

- 284 807 транзакций
- 492 мошеннические транзакции
- 30 признаков
- 1 целевой столбец (`Class`)

Значения целевого столбца:

- `0` = обычная транзакция
- `1` = мошенническая транзакция

Датасет является сильно несбалансированным, так как мошеннические операции составляют очень небольшую часть всех транзакций.

## Используемые технологии

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab

## Этапы проекта

1. Загрузка и изучение датасета
2. Первичный анализ данных
3. Разделение признаков и целевой переменной
4. Разделение данных на обучающую и тестовую выборки
5. Обучение модели Random Forest
6. Получение предсказаний
7. Оценка качества модели
8. Визуализация результатов с помощью матрицы ошибок

## Конвейер машинного обучения (Machine Learning Pipeline)

### Подготовка данных

Целевой столбец (`Class`) отделяется от признаков:

```python
X = df.drop("Class", axis=1)
Y = df["Class"]
```

### Разделение данных

Датасет разделяется на обучающую и тестовую выборки:

```python
train_test_split(X, Y, test_size=0.2, random_state=42)
```

### Обучение модели

Используется модель Random Forest:

```python
RandomForestClassifier(n_estimators=100, random_state=42)
```

### Оценка качества

Модель оценивается с помощью следующих метрик:

- Accuracy (точность)
- Recall (полнота)
- Confusion Matrix (матрица ошибок)

Так как датасет сильно несбалансирован, Recall является одной из наиболее важных метрик.

## Что я изучил

В ходе выполнения проекта я научился:

- Работать с DataFrame в Pandas
- Анализировать и исследовать данные
- Разделять данные на обучающую и тестовую выборки
- Обучать модели машинного обучения
- Делать предсказания с помощью Scikit-learn
- Оценивать качество модели
- Понимать различия между Accuracy и Recall
- Визуализировать результаты с помощью матрицы ошибок

## Результаты

Пример метрик:

- Accuracy: 0.9995611109160493
- Recall: 0.7653061224489796

## Возможные улучшения

В будущих версиях проекта можно добавить:

- Feature Engineering (создание новых признаков)
- Подбор гиперпараметров модели
- Кросс-валидацию
- Сравнение нескольких моделей
- Более эффективную работу с несбалансированными данными

## Автор

Это мой первый проект по машинному обучению и часть моего пути к профессии AI Engineer.

English: 

# Credit Card Fraud Detection

My first machine learning project.

## Project Goal

The goal of this project is to build a machine learning model that can detect fraudulent credit card transactions.

## Dataset

This project uses the Credit Card Fraud Detection dataset from Kaggle.

Dataset information:

- 284,807 transactions
- 492 fraudulent transactions
- 30 features
- 1 target column (`Class`)

Target values:

- `0` = legitimate transaction
- `1` = fraudulent transaction

The dataset is highly imbalanced because fraudulent transactions represent only a very small percentage of all operations.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab

## Project Workflow

1. Load and inspect the dataset
2. Perform basic data analysis
3. Separate features and target variable
4. Split the data into training and testing sets
5. Train a Random Forest classifier
6. Generate predictions
7. Evaluate model performance
8. Visualize results using a confusion matrix

## Machine Learning Pipeline

### Data Preparation

The target column (`Class`) is separated from the input features:

```python
X = df.drop("Class", axis=1)
Y = df["Class"]
```

### Train-Test Split

The dataset is divided into training and testing sets:

```python
train_test_split(X, Y, test_size=0.2, random_state=42)
```

### Model Training

A Random Forest classifier is used:

```python
RandomForestClassifier(n_estimators=100, random_state=42)
```

### Evaluation

The model is evaluated using:

- Accuracy
- Recall
- Confusion Matrix

Because the dataset is highly imbalanced, Recall is one of the most important metrics.

## What I Learned

During this project I learned how to:

- Work with Pandas DataFrames
- Explore and analyze datasets
- Split data into training and testing sets
- Train a machine learning model
- Make predictions using Scikit-learn
- Evaluate model performance
- Understand the difference between Accuracy and Recall
- Visualize results with a confusion matrix

## Results

Example metrics:

- Accuracy: 0.9995611109160493
- Recall: 0.7653061224489796

## Future Improvements

Possible improvements for future versions:

- Feature engineering
- Hyperparameter tuning
- Cross-validation
- Comparison with other models
- Better handling of class imbalance

## Author

scallym

This is my first machine learning project and part of my journey toward becoming an AI Engineer.
