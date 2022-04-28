import numpy as np

'''Policy Risk and epsilon_ATT employ when the couterfactual data is not available. On the Other hand epsilon_ATE and epsilon_PEHE use when the counterfactual and factual outcomes are available.'''


# Absolute error for the Average Treatment Effect on the Treated (epsilon_ATT)
def epsilon_att(pred_effect, outcome, treatment, experiment):
    
    ep_att = np.abs((np.mean(outcome[treatment>0]) - np.mean(outcome[(1-treatment+experiment)>1])) - np.mean(pred_effect[(treatment+experiment)>1]))
    
    return ep_att


# Error based on the Precision in Estimation of Heterogeneous Effect (epsilon_PEHE) or RMSE for Individual Treatment Effect (ITE)
def epsilon_pehe(true_ind_effect, pred_ind_effect):

    ep_pehe = np.sqrt(np.mean((pred_ind_effect - true_ind_effect)**2))
    
    return ep_pehe


# Absolute error for the Average Treatment Effect (epsilon_ATE)
def epsilon_ate(true_ind_effect, pred_ind_effect):
    
    ep_ate = np.abs(np.mean(pred_ind_effect) - np.mean(true_ind_effect))
    
    return ep_ate



# This functions have been provided by using D. Machlanski github rep: https://github.com/dmachlanski
# Policy Risk (R_pol)
def policy_risk(pred_effect, outcome, treatment, experiment):

    # the case of e > 0
    t_e = treatment[experiment > 0]
    y_e = outcome[experiment > 0]
    pred_effect_e = pred_effect[experiment > 0]

    if np.any(np.isnan(pred_effect_e)):
        return np.nan

    policy = pred_effect_e > 0.0
    treat_overlap = (policy == t_e) * (t_e > 0)
    control_overlap = (policy == t_e) * (t_e < 1)

    if np.sum(treat_overlap) == 0:
        treat_value = 0
    else:
        treat_value = np.mean(y_e[treat_overlap])

    if np.sum(control_overlap) == 0:
        control_value = 0
    else:
        control_value = np.mean(y_e[control_overlap])

    pit = np.mean(policy)
    policy_value = pit * treat_value + (1.0 - pit) * control_value

    return 1.0 - policy_value
