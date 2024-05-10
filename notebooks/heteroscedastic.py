import torch
import math


def _gamma(s):
    return math.gamma(s)


def _radius(n, alpha):
    """Return radius R for a given dimension n and alpha."""
    return ((_gamma(n/2 + alpha/(alpha-1)) /
             (_gamma(alpha/(alpha-1)) * math.pi**(n/2))) *
            (2 / (alpha-1)) ** (1/(alpha-1))) ** ((alpha-1)/(2 + (alpha-1)*n))


def squared_loss_with_sigma(prediction, sigma_sq, target, alpha=1):
    """Compute the squared loss when the model returns a variance in addition to the mean."""
    # The first output is the mean, the second is log_sigma_sq.
    if alpha == 1:
        # Gaussian case.
        loss = torch.mean((prediction - target)**2 / (2 * sigma_sq) + torch.log(sigma_sq) / 2)
    else:
        # Beta-Gaussian.
        R = _radius(1, alpha)
        loss = torch.mean((prediction - target)**2 / (2 * sigma_sq)
                          + 1/(alpha*(alpha-1))
                          #  - (R**2 * (1 + alpha) / (6*alpha - 2)) *  torch.exp(log_sigma_sq * (1 - alpha) / (1 + alpha)))
                          - ((R**2) * (1+alpha) / (6*alpha-2)) * sigma_sq ** ((1-alpha)/(1+alpha))
                          # - (R**2 / (3*alpha - 1)) * sigma_sq**((1 - alpha) / (1 + alpha))
                         )
    return loss


class BetaGaussianModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def loss(self, x, y):
        if self.uncertainty:
            y_pred, sigma_sq = self(x)
            lv = squared_loss_with_sigma(y_pred, sigma_sq, y,
                                         alpha=self.alpha)
        else:
            y_pred = self(x)
            mse_obj =  torch.nn.MSELoss()
            lv =  mse_obj(y_pred, y)
        return lv


def log_softplus(x):
    return torch.nn.functional.softplus(x).log()


class BetaGaussianLM(BetaGaussianModel):
    def __init__(self, input_size, uncertainty=False,
                 bias_y=True,
                 heteroscedastic=False, alpha=1.0):
        super().__init__()
        self.input_size = input_size
        self.uncertainty = uncertainty
        self.heteroscedastic = heteroscedastic
        self.alpha = alpha

        self.pred_y = torch.nn.Linear(self.input_size, 1, bias=bias_y)

        if self.heteroscedastic:
            assert(self.uncertainty)
            self.pred_sigma_sq = torch.nn.Linear(self.input_size, 1, bias=True)
            self.pred_sigma_sq.weight.data[:] = self.pred_sigma_sq.weight.data ** 2
            self.pred_sigma_sq.bias.data[:] = self.pred_sigma_sq.bias.data ** 2

        elif self.uncertainty:
            self.sigma_sq = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        pred_y = self.pred_y(x).squeeze()

        if self.uncertainty:
            if self.heteroscedastic:
                pred_sigma_sq = self.pred_sigma_sq(x).squeeze()
            else:
                pred_sigma_sq = self.sigma_sq.repeat(pred_y.shape[0])

            # pred_log_sigma_sq = log_softplus(pred_log_sigma_sq)
            # pred_sigma_sq = torch.clip(pred_sigma_sq, min=0.01, max=100)
            
            
            # pred_sigma_sq = torch.nn.functional.softplus(pred_sigma_sq)
            pred_sigma_sq = pred_sigma_sq ** 2  # torch.clip(pred_sigma_sq, min=0)
            
            # pred_sigma_sq = torch.clip(pred_sigma_sq, max=200)
            # pred_sigma_sq = 100 * torch.sigmoid(pred_sigma_sq)
            return pred_y, pred_sigma_sq
        else:
            return pred_y

    def predict(self, x):
        return self.pred_y(x).squeeze()


class BetaGaussianMLP(BetaGaussianModel):
    """
    OUTDATED at the moment
    Simple MLP for regression with one hidden layer.
    If uncertainty=True, estimate the variance in addition to the mean.
    If heteroscedastic=False, the variance is constant and it is learned as a parameter.
    If heteroscedastic=True, the variance is input-dependent and it is a second output of the MLP.
    In fact, what is returned is the log of the variance which is unconstrained (can be positive or negative).
    """
    def __init__(self, input_size, hidden_size, dropout_rate, uncertainty=False,
                 heteroscedastic=False, alpha=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.uncertainty = uncertainty
        self.heteroscedastic = heteroscedastic
        self.alpha = alpha

        if self.heteroscedastic:
            assert(self.uncertainty)
            # The first output is the mean, the second is log_sigma_sq.
            self.fc2 = torch.nn.Linear(self.hidden_size, 2)

        elif self.uncertainty:
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            self.log_sigma_sq = torch.nn.Parameter(torch.randn(1))

        else:
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        hidden = self.act(self.fc1(x))
        output = self.fc2(self.dropout(hidden))

        pred_y = output[:, 0]
        if self.uncertainty:
            if self.heteroscedastic:
                pred_log_sigma_sq = output[:, 1]
            else:
                pred_log_sigma_sq = self.log_sigma_sq.repeat(output.shape[0])
            
            pred_log_sigma_sq = torch.clip(pred_log_sigma_sq, max=100)
            # pred_log_sigma_sq = log_softplus(pred_log_sigma_sq)
            return pred_y, pred_log_sigma_sq

        else:
            return pred_y

    def predict(self, x):
        hidden = self.act(self.fc1(x))
        output = self.fc2(self.dropout(hidden))
        return output[:, 0]
