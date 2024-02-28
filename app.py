# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------

import altair as alt
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# ----------------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------------

__author__ = "Markku Laine"
__email__ = "markku.laine@gmail.com"
__version__ = "1.0"


# ----------------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------------

# Constants
DATASETS_DIR = Path("datasets")


def main():
    initialize_session()
    render_ui()


def initialize_session():
    # if "preview_data" not in st.session_state:
    #     st.session_state["preview_data"] = False

    if "show_evaluation_results" not in st.session_state:
        st.session_state["show_evaluation_results"] = False

    if "display_results" not in st.session_state:
        st.session_state["display_results"] = False

    if "metrics" not in st.session_state:
        st.session_state["metrics"] = {
            "previous_scores" : {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None
            },
            "present_scores" : {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None
            }
        }

    if "y_test" not in st.session_state:
        st.session_state["y_test"] = None
    if "y_pred" not in st.session_state:
        st.session_state["y_pred"] = None
    if "model" not in st.session_state:
        st.session_state["model"] = None


def render_ui():
    # Main
    st.title("Interactive Employee Classifier")
    st.write("A brief introduction.")
    dataset_preview_container = st.container()


    # Sidebar
    with st.sidebar:
        # Header
        st.image(image="assets/img/streamlit_logo.svg", width=200)
        st.title(f"Interactive Employee Classifier `version {__version__}`")

        # Dataset
        st.header("Dataset")
        dataset = st.selectbox(label="Choose CSV file", options=("hr_dataset_v1.csv", "hr_dataset_v2.csv"), index=0, key="dataset")
        df = load_data(DATASETS_DIR / str(dataset))
        X_train, X_test, y_train, y_test = split_data(df)
        if st.checkbox(label="Preview dataset", value=False, key="preview_dataset"):
            with dataset_preview_container:
                st.header("Dataset Preview")
                st.dataframe(data=df, use_container_width=True)

        # Model
        st.header("Model")
        algorithm = st.selectbox(label="Choose algorithm", options=("Decision Tree", "Random Forest", "XGBoost"), index=0, key="algorithm")

        # Form
        form = st.form(key="data_modeling_form", border=False)
        with form:
            # Hyperparameters
            st.header("Hyperparameters")
            if algorithm == "Decision Tree":
                max_depth = st.number_input(label="Max depth", help="The maximum depth of the tree.", min_value=1, max_value=None, value=None, step=1, placeholder="None")
                min_samples_leaf = st.number_input(label="Min samples leaf", help="The minimum number of samples required to be at a leaf node.", min_value=1, max_value=None, value=1, step=1, placeholder="1")
                min_samples_split = st.number_input(label="Min samples split", help="The minimum number of samples required to split an internal node.", min_value=2, max_value=None, value=2, step=1, placeholder="2")
            elif algorithm == "Random Forest":
                max_depth = st.number_input(label="Max depth", help="The maximum depth of the tree.", min_value=1, max_value=None, value=None, step=1, placeholder="None")
                max_features = st.number_input(label="Max features", help="The number of features to consider when looking for the best split.", min_value=1, max_value=None, value=None, step=1, placeholder="None")
                max_samples = st.number_input(label="Max samples", help="If bootstrap is True, the number of samples to draw from X to train each base estimator.", min_value=1, max_value=X_train.shape[0], value=None, step=1, placeholder="None")
                min_samples_leaf = st.number_input(label="Min samples leaf", help="The minimum number of samples required to be at a leaf node.", min_value=1, max_value=None, value=1, step=1, placeholder="1")
                min_samples_split = st.number_input(label="Min samples split", help="The minimum number of samples required to split an internal node.", min_value=2, max_value=None, value=2, step=1, placeholder="2")
                n_estimators = st.number_input(label="Number of estimators", help="The number of trees in the forest.", min_value=1, max_value=500, value=100, step=1, placeholder="100")
            elif algorithm == "XGBoost":
                colsample_bytree = st.number_input(label="Colsample bytree", help="Subsample ratio of columns when constructing each tree.", min_value=0.0, max_value=1.0, value=None, step=0.1, placeholder="None")
                learning_rate = st.number_input(label="Learning rate", help='Boosting learning rate (xgb’s “eta”)', min_value=0.0, max_value=1.0, value=None, step=0.1, placeholder="None")
                max_depth = st.number_input(label="Max depth", help="Maximum tree depth for base learners.", min_value=1, max_value=None, value=None, step=1, placeholder="None")
                min_child_weight = st.number_input(label="Min child weight", help="Minimum sum of instance weight(hessian) needed in a child.", min_value=0.0, max_value=1.0, value=None, step=0.1, placeholder="None")
                n_estimators = st.number_input(label="Number of estimators", help="Number of boosting rounds.", min_value=1, max_value=500, value=100, step=1, placeholder="100")
                subsample = st.number_input(label="Subsample", help="Subsample ratio of the training instance.", min_value=0.0, max_value=1.0, value=1.0, step=0.1, placeholder="1.0")

            # Button
            submitted = st.form_submit_button(label="Run model", type="primary", use_container_width=True)

        if submitted:
            st.session_state["show_evaluation_results"] = True

            # Create model instance
            if algorithm == "Decision Tree":
                model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, random_state=42)
            elif algorithm == "Random Forest":
                model = RandomForestClassifier(max_depth=max_depth, max_features=max_features, max_samples=max_samples, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators, n_jobs=-1, random_state=42)
            elif algorithm == "XGBoost":
                model = XGBClassifier(colsample_bytree=colsample_bytree, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight, n_estimators=n_estimators, subsample=subsample, objective="binary:logistic", n_jobs=-1, random_state=42)

            # Fit model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Update session state
            st.session_state["y_test"] = y_test
            st.session_state["y_pred"] = y_pred
            st.session_state["model"] = model
            st.session_state["metrics"]["previous_scores"]["accuracy"] = st.session_state["metrics"]["present_scores"]["accuracy"]
            st.session_state["metrics"]["previous_scores"]["precision"] = st.session_state["metrics"]["present_scores"]["precision"]
            st.session_state["metrics"]["previous_scores"]["recall"] = st.session_state["metrics"]["present_scores"]["recall"]
            st.session_state["metrics"]["previous_scores"]["f1"] = st.session_state["metrics"]["present_scores"]["f1"]
            st.session_state["metrics"]["present_scores"]["accuracy"] = accuracy_score(y_test, y_pred)
            st.session_state["metrics"]["present_scores"]["precision"] = precision_score(y_test, y_pred)
            st.session_state["metrics"]["present_scores"]["recall"] = recall_score(y_test, y_pred)
            st.session_state["metrics"]["present_scores"]["f1"] = f1_score(y_test, y_pred)

        # Footer
        st.markdown("""
            ---
            Created with ❤️ by [Markku Laine](https://markkulaine.com).            
        """)

    # Show results
    if st.session_state["show_evaluation_results"]:
        show_evaluation_results()



    # # Model hyperparameters
    # form.subheader("Model Hyperparameters")
    # if classifier == "Decision Tree":
    #     max_depth = form.number_input(label="Max depth", help="The maximum depth of the tree.", min_value=1, max_value=None, value=None, step=1, placeholder="None")
    #     min_samples_leaf = form.number_input(label="Min samples leaf", help="The minimum number of samples required to be at a leaf node.", min_value=1, max_value=None, value=1, step=1, placeholder="1")
    #     min_samples_split = form.number_input(label="Min samples split", help="The minimum number of samples required to split an internal node.", min_value=2, max_value=None, value=2, step=1, placeholder="2")
    # elif classifier == "Random Forest":
    #     max_depth = form.number_input(label="Max depth", help="The maximum depth of the tree.", min_value=1, max_value=None, value=None, step=1, placeholder="None")
    #     max_features = form.number_input(label="Max features", help="The number of features to consider when looking for the best split.", min_value=1, max_value=None, value=None, step=1, placeholder="None")
    #     max_samples = form.number_input(label="Max samples", help="If bootstrap is True, the number of samples to draw from X to train each base estimator.", min_value=1, max_value=X_train.shape[0], value=None, step=1, placeholder="None")
    #     min_samples_leaf = form.number_input(label="Min samples leaf", help="The minimum number of samples required to be at a leaf node.", min_value=1, max_value=None, value=1, step=1, placeholder="1")
    #     min_samples_split = form.number_input(label="Min samples split", help="The minimum number of samples required to split an internal node.", min_value=2, max_value=None, value=2, step=1, placeholder="2")
    #     n_estimators = form.number_input(label="Number of estimators", help="The number of trees in the forest.", min_value=1, max_value=500, value=100, step=1, placeholder="100")
    # elif classifier == "XGBoost":
    #     colsample_bytree = form.number_input(label="Colsample bytree", help="Subsample ratio of columns when constructing each tree.", min_value=0.0, max_value=1.0, value=None, step=0.1, placeholder="None")
    #     learning_rate = form.number_input(label="Learning rate", help='Boosting learning rate (xgb’s “eta”)', min_value=0.0, max_value=1.0, value=None, step=0.1, placeholder="None")
    #     max_depth = form.number_input(label="Max depth", help="Maximum tree depth for base learners.", min_value=1, max_value=None, value=None, step=1, placeholder="None")
    #     min_child_weight = form.number_input(label="Min child weight", help="Minimum sum of instance weight(hessian) needed in a child.", min_value=0.0, max_value=1.0, value=None, step=0.1, placeholder="None")
    #     n_estimators = form.number_input(label="Number of estimators", help="Number of boosting rounds.", min_value=1, max_value=500, value=100, step=1, placeholder="100")
    #     subsample = form.number_input(label="Subsample", help="Subsample ratio of the training instance.", min_value=0.0, max_value=1.0, value=1.0, step=0.1, placeholder="1.0")

    # # Model evaluation
    # form.subheader("Model Evaluation")
    # metrics = form.multiselect(label="Metrics", help="The metrics to measure classification performance.", options=("Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Feature Importances"), default=("Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Feature Importances"), placeholder="Choose metrics")


@st.cache_data
def load_data(filepath: Path) -> pd.DataFrame:
    # Load the dataset
    data = pd.read_csv(filepath)

    # Return the DataFrame
    return data


@st.cache_data
def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Select predictor (X) and target (y) variables
    X = df.drop(["left"], axis=1)
    y = df["left"]

    # Split into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Return the sets
    return X_train, X_test, y_train, y_test


def show_evaluation_results():
    st.header("Evaluation Results")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Metrics", "Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Feature Importances"])

    with tab1:
        plot_metrics()

    with tab2:
        plot_confusion_matrix()

    with tab3:
        plot_roc_curve()

    with tab4:
        plot_precision_recall_curve()

    with tab5:
        plot_feature_importances()


def plot_metrics():
    st.subheader("Metrics")

    # Retrieve session state
    previous_accuracy = st.session_state["metrics"]["previous_scores"]["accuracy"]
    previous_precision = st.session_state["metrics"]["previous_scores"]["precision"]
    previous_recall = st.session_state["metrics"]["previous_scores"]["recall"]
    previous_f1 = st.session_state["metrics"]["previous_scores"]["f1"]
    present_accuracy = st.session_state["metrics"]["present_scores"]["accuracy"]
    present_precision = st.session_state["metrics"]["present_scores"]["precision"]
    present_recall = st.session_state["metrics"]["present_scores"]["recall"]
    present_f1 = st.session_state["metrics"]["present_scores"]["f1"]

    # Compute deltas
    accuracy_delta = None if previous_accuracy is None else "{:.4f}".format(present_accuracy - previous_accuracy)
    precision_delta = None if previous_precision is None else "{:.4f}".format(present_precision - previous_precision)
    recall_delta = None if previous_recall is None else "{:.4f}".format(present_recall - previous_recall)
    f1_delta = None if previous_f1 is None else "{:.4f}".format(present_f1 - previous_f1)

    # Render metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Accuracy score", value=f"{present_accuracy:.4f}", delta=accuracy_delta)
    col2.metric(label="Precision score", value=f"{present_precision:.4f}", delta=precision_delta)
    col3.metric(label="Recall score", value=f"{present_recall:.4f}", delta=recall_delta)
    col4.metric(label="F1 score", value=f"{present_f1:.4f}", delta=f1_delta)


def plot_confusion_matrix():
    st.subheader("Confusion Matrix")

    # Retrieve session state
    y_test = st.session_state["y_test"]
    y_pred = st.session_state["y_pred"]

    # Update labels
    labels = ["Stayed", "Left"]
    label_mapping = {0: labels[0], 1: labels[1]}
    y_test_transformed = [label_mapping[label] for label in y_test]
    y_pred_transformed = [label_mapping[label] for label in y_pred]

    # Compute confusion matrix
    cm = confusion_matrix(y_test_transformed, y_pred_transformed, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # Render confusion matrix
    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d", cmap="Reds")
    st.pyplot(fig)


def plot_roc_curve():
    st.subheader("ROC Curve")

    # Retrieve session state
    y_test = st.session_state["y_test"]
    y_pred = st.session_state["y_pred"]
    algorithm = st.session_state["algorithm"]

    # Update labels
    labels = ["Stayed", "Left"]
    label_mapping = {0: labels[0], 1: labels[1]}
    y_test_transformed = [label_mapping[label] for label in y_test]

    # Compute ROC curve
    disp = RocCurveDisplay.from_predictions(y_true=y_test_transformed, y_pred=y_pred, name=algorithm, pos_label=labels[1])

    # Render ROC curve
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)


def plot_precision_recall_curve():
    st.subheader("Precision-Recall Curve")

    # Retrieve session state
    y_test = st.session_state["y_test"]
    y_pred = st.session_state["y_pred"]
    algorithm = st.session_state["algorithm"]

    # Update labels
    labels = ["Stayed", "Left"]
    label_mapping = {0: labels[0], 1: labels[1]}
    y_test_transformed = [label_mapping[label] for label in y_test]

    # Compute Precision-Recall curve
    disp = PrecisionRecallDisplay.from_predictions(y_true=y_test_transformed, y_pred=y_pred, name=algorithm, pos_label=labels[1])

    # Render Precision-Recall curve
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)


def plot_feature_importances():
    st.subheader("Feature Importances")

    # Retrieve session state
    model = st.session_state["model"]

    # Create feature importances dataframe
    feature_names = model.feature_names_in_
    feature_importances = model.feature_importances_
    df_fi = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances}).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    chart = alt.Chart(df_fi).mark_bar(color=st.get_option("theme.primaryColor")).encode(x="Importance", y=alt.Y("Feature", sort="-x", axis=alt.Axis(labelLimit=300)))
    st.altair_chart(chart, theme="streamlit", use_container_width=True)


if __name__ == '__main__':
    st.set_page_config(page_title="Interactive Employee Classifier", initial_sidebar_state="expanded")
    main()
