import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def predict_batch(table, model, features, threshold):
    result = table.copy()
    result['probability'] = model.predict_proba(table[features])[:, 1]
    result['prediction'] = (result['probability'] >= threshold).astype(int)
    return result


model = load_model()
config = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))
test = load_test()
features = test.drop(columns='target').columns.tolist()
X_test = test[features]
y_test = test['target']
probabilities = test_proba(model, X_test)

labels = {
    'fixed_acidity': 'фиксированная кислотность',
    'volatile_acidity': 'летучая кислотность',
    'citric_acid': 'лимонная кислота',
    'residual_sugar': 'остаточный сахар',
    'chlorides': 'хлориды',
    'free_sulfur_dioxide': 'свободный диоксид серы',
    'total_sulfur_dioxide': 'общий диоксид серы',
    'density': 'плотность',
    'pH': 'pH',
    'sulphates': 'сульфаты',
    'alcohol': 'алкоголь',
}

st.title('Подборка вин')
threshold = st.sidebar.slider(
    'порог', 0.0, 1.0, value=float(config['threshold']), step=0.001)

single_tab, metrics_tab, batch_tab, weights_tab = st.tabs([
    'Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

with single_tab:
    values = {}
    for feature in features:
        if feature != 'is_red':
            values[feature] = st.number_input(
                labels[feature],
                min_value=float(X_test[feature].min()),
                max_value=float(X_test[feature].max()),
                value=float(X_test[feature].median()),
                format='%.5f' if feature == 'density' else '%.3f')
    values['is_red'] = int(st.toggle(
        'красное вино (выключено — белое)',
        value=bool(X_test['is_red'].median())))
    sample = pd.DataFrame([values], columns=features)
    probability = float(model.predict_proba(sample)[0, 1])
    st.metric('вероятность хорошего вина', f'{probability:.1%}')
    if probability >= threshold:
        st.write('включить в подборку')
    else:
        st.write('не включать в подборку')

with metrics_tab:
    predictions = (probabilities >= threshold).astype(int)
    precision = metrics.precision_score(y_test, predictions, zero_division=0)
    recall = metrics.recall_score(y_test, predictions, zero_division=0)
    f1 = metrics.f1_score(y_test, predictions, zero_division=0)
    st.metric('precision', f'{precision:.3f}')
    st.metric('recall', f'{recall:.3f}')
    st.metric('F1', f'{f1:.3f}')

    cm = metrics.confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, ax = plt.subplots()
    metrics.ConfusionMatrixDisplay(cm).plot(ax=ax)
    ax.set_title('матрица ошибок')
    st.pyplot(fig)
    plt.close(fig)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    fig, ax = plt.subplots()
    metrics.RocCurveDisplay.from_predictions(y_test, probabilities, ax=ax)
    ax.scatter(fpr, recall, label='текущий порог')
    ax.set_title('ROC-кривая')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    metrics.PrecisionRecallDisplay.from_predictions(y_test, probabilities, ax=ax)
    pr_precision = precision if predictions.any() else 1.0
    ax.scatter(recall, pr_precision, label='текущий порог')
    ax.set_title('PR-кривая')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.write('при повышении порога подборка становится меньше: обычно растет точность, '
             'но больше хороших вин пропускается. при снижении порога подборка расширяется, '
             'но в нее попадает больше посредственных вин')

with batch_tab:
    uploaded = st.file_uploader('CSV с признаками вин без target', type=['csv'])
    if uploaded is not None:
        table = pd.read_csv(uploaded)
        missing = [column for column in features if column not in table.columns]
        if missing:
            st.error('не хватает столбцов: ' + ', '.join(missing))
        else:
            result = predict_batch(table, model, features, threshold)
            st.dataframe(result)
            st.download_button(
                'скачать предсказания', result.to_csv(index=False).encode('utf-8-sig'),
                file_name='wine_predictions.csv', mime='text/csv')

with weights_tab:
    coefficients = pd.Series(
        model.named_steps['logisticregression'].coef_[0], index=features).sort_values()
    fig, ax = plt.subplots()
    ax.barh(coefficients.index, coefficients.values)
    ax.set_xlabel('коэффициент после масштабирования признаков')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    positive = coefficients[coefficients > 0].nlargest(3).index.tolist()
    negative = coefficients[coefficients < 0].nsmallest(3).index.tolist()
    st.write('сильнее всего за хорошее вино: ' + ', '.join(positive))
    st.write('сильнее всего против: ' + ', '.join(negative))
