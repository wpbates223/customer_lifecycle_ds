def recommend_action(churn_prob, balance):
    if churn_prob > 0.7 and balance > 100000:
        return "Offer premium retention package"
    elif churn_prob > 0.5:
        return "Send targeted engagement email"
    else:
        return "Upsell credit product"