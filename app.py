import pandas as pd
import pathlib
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# Constants
DATASETS_DIR = pathlib.Path("datasets")

# Variables
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


@st.cache_data
def load_data(filepath) -> pd.DataFrame:
    # Load the dataset
    data = pd.read_csv(filepath)

    # Return the DataFrame
    return data


@st.cache_data
def split_data(df):
    # Select predictor (X) variables
    X = df.drop(["left"], axis=1)

    # Select target (y) variable
    y = df["left"]

    # Split into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Verify set sizes
    # print(f"Training set:   {len(X_train)/len(X):.2%} {X_train.shape} & {y_train.shape}")
    # print(f"Test set:       {len(X_test)/len(X):.2%} {X_test.shape} & {y_test.shape}")

    # Return the sets
    return X_train, X_test, y_train, y_test


def render_main_ui():
    st.title("Interactive Employee Classifier")


def render_sidebar_ui():
    st.sidebar.title("Settings")

    # Dataset
    st.sidebar.subheader("Dataset")
    dataset = st.sidebar.selectbox(label="CSV file", help="The HR dataset.", options=("hr_dataset_v1.csv", "hr_dataset_v2.csv"), index=0, key="dataset", placeholder="Choose a dataset")

    # Load data
    df = load_data(DATASETS_DIR / str(dataset))

    # Show data
    if st.sidebar.checkbox("Show data", False):
        st.subheader("Dataset")
        st.write(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    

    # Model
    st.sidebar.subheader("Model")
    classifier = st.sidebar.selectbox(label="Classifier", help="The classification model.", options=("Decision Tree", "Random Forest", "XGBoost"), index=0, key="classifier", placeholder="Choose a classifier")

    # Form
    form = st.sidebar.form(key="settings_form", border=False)

    # Model hyperparameters
    form.subheader("Model Hyperparameters")
    if classifier == "Decision Tree":
        max_depth = form.number_input(label="max_depth", help="The maximum depth of the tree.", min_value=1, max_value=None, value=None, step=1, placeholder="None")
        min_samples_leaf = form.number_input(label="min_samples_leaf", help="The minimum number of samples required to be at a leaf node.", min_value=1, max_value=None, value=1, step=1, placeholder="1")
        min_samples_split = form.number_input(label="min_samples_split", help="The minimum number of samples required to split an internal node.", min_value=2, max_value=None, value=2, step=1, placeholder="2")
    elif classifier == "Random Forest":
        max_depth = form.number_input(label="max_depth", help="The maximum depth of the tree.", min_value=1, max_value=None, value=None, step=1, placeholder="None")
        max_features = form.number_input(label="max_features", help="The number of features to consider when looking for the best split.", min_value=1, max_value=None, value=None, step=1, placeholder="None")
        max_samples = form.number_input(label="max_samples", help="If bootstrap is True, the number of samples to draw from X to train each base estimator.", min_value=1, max_value=X_train.shape[0], value=None, step=1, placeholder="None")
        min_samples_leaf = form.number_input(label="min_samples_leaf", help="The minimum number of samples required to be at a leaf node.", min_value=1, max_value=None, value=1, step=1, placeholder="1")
        min_samples_split = form.number_input(label="min_samples_split", help="The minimum number of samples required to split an internal node.", min_value=2, max_value=None, value=2, step=1, placeholder="2")
        n_estimators = form.number_input(label="n_estimators", help="The number of trees in the forest.", min_value=1, max_value=500, value=100, step=1, placeholder="100")
    elif classifier == "XGBoost":
        colsample_bytree = form.number_input(label="colsample_bytree", help="Subsample ratio of columns when constructing each tree.", min_value=0.0, max_value=1.0, value=None, step=0.1, placeholder="None")
        learning_rate = form.number_input(label="learning_rate", help='Boosting learning rate (xgb’s “eta”)', min_value=0.0, max_value=1.0, value=None, step=0.1, placeholder="None")
        max_depth = form.number_input(label="max_depth", help="Maximum tree depth for base learners.", min_value=1, max_value=None, value=None, step=1, placeholder="None")
        min_child_weight = form.number_input(label="min_child_weight", help="Minimum sum of instance weight(hessian) needed in a child.", min_value=0.0, max_value=1.0, value=None, step=0.1, placeholder="None")
        n_estimators = form.number_input(label="n_estimators", help="Number of boosting rounds.", min_value=1, max_value=500, value=100, step=1, placeholder="100")
        subsample = form.number_input(label="subsample", help="Subsample ratio of the training instance.", min_value=0.0, max_value=1.0, value=1.0, step=0.1, placeholder="1.0")

    # Model evaluation
    form.subheader("Model Evaluation")
    metrics = form.multiselect(label="Metrics", help="The metrics to measure classification performance.", options=("Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Feature Importances"), default=("Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Feature Importances"), placeholder="Choose metrics")

    submitted = form.form_submit_button(label="Classify", type="primary", use_container_width=True)
    if submitted:
        # Create model instance
        if classifier == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, random_state=42)
        elif classifier == "Random Forest":
            model = RandomForestClassifier(max_depth=max_depth, max_features=max_features, max_samples=max_samples, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators, n_jobs=-1, random_state=42)
        elif classifier == "XGBoost":
            model = XGBClassifier(colsample_bytree=colsample_bytree, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight, n_estimators=n_estimators, n_jobs=-1, objective="binary:logistic", random_state=42, subsample=subsample)

        # Fit model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Update previous metric scores
        st.session_state["metrics"]["previous_scores"]["accuracy"] = st.session_state["metrics"]["present_scores"]["accuracy"]
        st.session_state["metrics"]["previous_scores"]["precision"] = st.session_state["metrics"]["present_scores"]["precision"]
        st.session_state["metrics"]["previous_scores"]["recall"] = st.session_state["metrics"]["present_scores"]["recall"]
        st.session_state["metrics"]["previous_scores"]["f1"] = st.session_state["metrics"]["present_scores"]["f1"]

        # Update present metric scores
        st.session_state["metrics"]["present_scores"]["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
        st.session_state["metrics"]["present_scores"]["precision"] = round(precision_score(y_test, y_pred), 4)
        st.session_state["metrics"]["present_scores"]["recall"] = round(recall_score(y_test, y_pred), 4)
        st.session_state["metrics"]["present_scores"]["f1"] = round(f1_score(y_test, y_pred), 4)

        st.session_state["model"] = model
        st.session_state["y_test"] = y_test
        st.session_state["y_pred"] = y_pred
        st.session_state["display_results"] = True

    if st.session_state["display_results"]:
        display_results()


def display_results():
    model = st.session_state["model"]
    y_test = st.session_state["y_test"]
    y_pred = st.session_state["y_pred"]

    # Evaluate model
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Metrics", "Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Feature Importances"])

    # Metrics
    with tab1:
        display_metrics()

    # Confusion matrix
    with tab2:
        display_confusion_matrix(y_test, y_pred, model.classes_)

    # ROC curve
    with tab3:
        display_roc_curve(y_test, y_pred)
    
    # Precision-Recall curve
    with tab4:
        display_precision_recall_curve(y_test, y_pred)
    
    # Feature importances
    with tab5:
        display_feature_importances(model)


def display_metrics():
    st.subheader("Metrics")

    # Retrieve previous metric scores
    previous_accuracy = st.session_state["metrics"]["previous_scores"]["accuracy"]
    previous_precision = st.session_state["metrics"]["previous_scores"]["precision"]
    previous_recall = st.session_state["metrics"]["previous_scores"]["recall"]
    previous_f1 = st.session_state["metrics"]["previous_scores"]["f1"]

    # Retrieve present metric scores
    present_accuracy = st.session_state["metrics"]["present_scores"]["accuracy"]
    present_precision = st.session_state["metrics"]["present_scores"]["precision"]
    present_recall = st.session_state["metrics"]["present_scores"]["recall"]
    present_f1 = st.session_state["metrics"]["present_scores"]["f1"]

    # Compute deltas
    accuracy_delta = None if previous_accuracy is None else "{:.4f}".format(present_accuracy - previous_accuracy)
    precision_delta = None if previous_precision is None else "{:.4f}".format(present_precision - previous_precision)
    recall_delta = None if previous_recall is None else "{:.4f}".format(present_recall - previous_recall)
    f1_delta = None if previous_f1 is None else "{:.4f}".format(present_f1 - previous_f1)

    # Show metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Accuracy score", help="Accuracy classification score.", value=f"{present_accuracy:.4f}", delta=accuracy_delta)
    col2.metric(label="Precision score", help="The precision is the ratio `tp / (tp + fp)` where `tp` is the number of true positives and `fp` the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.", value=f"{present_precision:.4f}", delta=precision_delta)
    col3.metric(label="Recall score", help="The recall is the ratio `tp / (tp + fn)` where `tp` is the number of true positives and `fn` the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.", value=f"{present_recall:.4f}", delta=recall_delta)
    col4.metric(label="F1 score", help="The F1 score can be interpreted as a harmonic mean of the precision and recall.", value=f"{present_f1:.4f}", delta=f1_delta)


def display_confusion_matrix(y_test, y_pred, labels):
    st.subheader("Confusion Matrix")

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # Show confusion matrix
    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d", cmap="Reds")
    st.pyplot(fig)


def display_roc_curve(y_test, y_pred):
    st.subheader("ROC Curve")

    # Compute ROC curve
    disp = RocCurveDisplay.from_predictions(y_test, y_pred)

    # Show ROC curve
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)


def display_precision_recall_curve(y_test, y_pred):
    st.subheader("Precision-Recall Curve")

    # Compute Precision-Recall curve
    disp = PrecisionRecallDisplay.from_predictions(y_test, y_pred)

    # Show Precision-Recall curve
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)


def display_feature_importances(model):
    # Create feature importances dataframe
    feature_names = model.feature_names_in_
    feature_importances = model.feature_importances_
    df_fi = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances}).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Define helper variables
    colors = ["#FF4B4B"]
    plot_color_1 = "#595959"
    plot_color_2 = "#F0F0F0"

    # Create a figure and subplots
    fig, ax = plt.subplots()

    # Create a bar plot
    sns.barplot(ax=ax, data=df_fi, x="Importance", y="Feature", color=colors[0], ec="none", alpha=1.0 ,saturation=1.0)

    # Customize axis spines
    sns.despine(trim=False)  # remove axis spines
    ax.spines[["top", "right", "bottom"]].set_visible(False)  # remove some borders
    ax.spines[["left"]].set_color(plot_color_1)  # set bottom border color
    ax.spines[["left"]].set_linewidth(1)  # set bottom border width

    # Customize ticks
    ax.tick_params(axis="x", which="both", bottom=False, top=False, colors=plot_color_1, labelcolor=plot_color_1)  # set x-axis ticks
    ax.tick_params(axis="y", which="both", left=True, right=False, colors=plot_color_1, labelcolor=plot_color_1)  # set y-axis ticks

    # Customize grid lines
    ax.xaxis.grid(color=plot_color_2)  # set y-axis grid line colors
    ax.set_axisbelow(True)  # show axes below

    # Customize labels
    ax.set_xlabel("Importance", color=plot_color_1, labelpad=8, fontsize=10, fontweight="normal")  # set x-axis label
    ax.set_ylabel("Feature", color=plot_color_1, labelpad=8, fontsize=10, fontweight="normal")  # set y-axis label

    # Show the plot
    st.pyplot(fig)


def main():
    # Render main UI
    render_main_ui()

    # Render sidebar UI
    render_sidebar_ui()


if __name__ == '__main__':
    main()
