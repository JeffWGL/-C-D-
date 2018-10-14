import numpy as np


class CD_Regressor(object):
    def __init__(self):
        """
        given the Cobb-Douglas utility function U = X1^α * X2^(1-α), we'll only need to specify one of the exponential
        powers; in this case, alpha.
        """
        self.params = None

    def get_params(self):
        """Returns the current value of self.params i.e. alpha"""
        return self.params

    def initialize_params(self, X, P):
        """
        X: quantities of health services consumed at market equilibrium of shape(N, 2);
        P: prices at market equilibrium of shape(N, 2);

        N indicates the number of bundles chosen, while the second dimension indicates the number of types of goods or
        services, which is, in this particular case, 2, as only first-class and second-class health services are
        considered;

        each row of X refers to a chosen bundle of services under a particular pair of prices given by the corresponding
        row of P.

        Returns: an updated value of parameter alpha, initialized based on the very implications of C-D utility model
        """
        # expenditure on first-class and second-class health services
        exps = X * P

        # budget constraints
        bgts = np.sum(exps, axis=1, keepdims=True)

        # the theoretical value of alpha given by mathematical deductions within C-D model, w.r.t N bundles chosen
        alphas = exps[:, 0].reshape((exps.shape[0], 1)) / bgts

        # initialize self.params with the average of alphas
        self.params = np.mean(alphas)

        return self.params

    def scores(self, X):
        """
        X: quantities of health services in chosen bundles of shape(N, 2)

        Returns: the C-D utilities U of shape(N, 1), corresponding to each bundle
        """
        U = np.ones((X.shape[0], 1))

        try:
            alpha = self.params
            U = (X[:, 0] ** alpha) * (X[:, 1] ** (1 - alpha))
            U = U.reshape((X.shape[0], 1))

        except ValueError:
            print("*** parameters might not yet have been initialized!! ***")

        return U

    def loss(self, X, P, reg=0):
        """
        X: quantities of health services in chosen bundles of shape(N, 2)
        P: prices of health services in chosen bundles of shape(N, 2)
        reg: regularization strength

        Returns: the loss defined according to the postulate of rationality, and the gradient with respect to alpha
        """
        loss = None
        grad = None

        N = X.shape[0]
        # compute the matrix "cost" of shape(N, N), where c(i,j) refers to the total cost of the ith bundle under the
        # jth pair of prices. for instance, cost(2,3) = x(2,1) * p(3,1) + x(2,2) * p(3,2). in particular, the diagonal
        # elements refers to the actual amounts of budget constraints.
        cost = np.dot(X, P.T)

        # compute the matrix "feasible" of shape(N, N), where each element equals to 1 if and only if the corresponding
        # cost of a particular bundle is less than the actual amount of budget constraint under the same prices.
        # i.e. feasible(i,j)==1 <--> cost(i,j)<=cost(j,j)
        budget = cost[np.arange(N), np.arange(N)]
        feasible = (cost <= budget).astype(int)

        # compute the matrix "delta_U" indicating the difference of utilities of shape(N, N).
        # concretely, delta_U(i,j) = U(i) - U(j), where U(k) refers to the utility of the kth bundle.
        U = self.scores(X)
        delta_U = np.hstack([U] * N) - U.T

        # compute the matrix "delta_U_eff" where each element of "delta_U" is passed through a max{0,x} gate.
        # i.e. only those whose utilities are higher than the actual one may count.
        sgn = (delta_U > 0).astype(int)
        delta_U_eff = delta_U * sgn

        # compute the matrix L where each effective difference of utilities is preserved provided that the corresponding
        # bundle is feasible, whereas in all other cases elements are left with zero.
        L = feasible * delta_U_eff

        # compute the final loss
        loss = np.sum(L) / N + reg * (self.params ** 2)

        # now we compute the gradient
        # note that the total loss = ΣL(i,j) = Σdelta_U(i,j) = Σ(U(i)- U(j)), over certain conditions. Meanwhile, it is
        # delighting that after due calculations of the analytical derivatives, we have:
        # dU(k)/dα = U * ln(x(k,1)/x(k,2)),
        # which facilitates the overall computation of the gradient and ?? the need of backpropagation.

        # compute the matrix dU/dα of shape(N, 1)
        dU = U * np.log(X[:, 0] / X[:, 1]).reshape((N, 1))

        # compute the matrix ddU (i.e. d(delta_U)/dα) of shape(N, N)
        ddU = np.hstack([dU] * N) - dU.T

        # compute the final gradient
        sgn = (L > 0).astype(int)
        grad = np.sum(ddU * sgn) / N + 2 * reg * self.params

        # Q: shall the regularization term be considered?
        return loss, grad

    def train(self, X, P, learning_rate=1e-3, reg=0, num_iters=1000, batch_size=200, verbose=False):
        """
        carry out the optimization process.

        X: quantities of health services in chosen bundles of shape(N, 2)
        P: prices of health services in chosen bundles of shape(N, 2)
        learning_rate: controls the "speed" of gradient descent
        reg: regularization strength
        num_iters: number of iterations during SGD
        batch_size: the set size of a batch of data within a single iteration
        verbose: the boolean which controls whether some details should be printed out together with the ongoing process

        Returns: loss_history (w.r.t. each iteration during SGD)
        """
        # initialize alpha
        self.initialize_params(X, P)

        N = X.shape[0]
        loss_history = []

        # optimization: SGD(stochastic gradient descent)
        for i in range(num_iters):
            indices = np.random.choice(N, batch_size, replace=True)
            X_batch = X[indices]
            P_batch = P[indices]

            loss, grad = self.loss(X_batch, P_batch, reg)
            loss_history.append(loss)
            self.params += -learning_rate * grad

            if verbose and i % 100 == 0:
                print('iteration %d / %d: loss %f' % (i, num_iters, loss))

        return loss_history

    def predict(self):
        """
        no need to predict anything, as we only want the value of alpha
        """
        pass
