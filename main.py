from data_loader import load_price_data
from features import build_feature_frame
from targets import add_classification_label
from models import prepare_dataset, train_regression_model, train_classification_model
from backtest import simulate_trades
import matplotlib.pyplot as plt

def main():
    df = load_price_data("data/aapl_daily.csv")
    df = build_feature_frame(df)
    df = add_classification_label(df, horizon=5, threshold=0.03)

    X, y_reg, y_cls = prepare_dataset(df, horizon=5)

    reg_model, reg_info = train_regression_model(X, y_reg)
    cls_model, cls_info = train_classification_model(X, y_cls)

    (_, X_test, _, y_test_reg, y_pred_reg, mae, r2) = reg_info
    (_, X_test_c, _, y_test_cls, y_pred_cls, y_proba_cls, acc, auc) = cls_info

    print(f"Regression MAE: {mae:.4f}, R2: {r2:.3f}")
    print(f"Classification ACC: {acc:.3f}, AUC: {auc:.3f}")

    df_test = df.iloc[-len(X_test):]
    bt_df = simulate_trades(df_test, y_pred_reg, y_proba_cls)

    plt.figure(figsize=(10,4))
    bt_df['equity'].plot()
    plt.title("Equity Curve")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
