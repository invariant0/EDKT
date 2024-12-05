import torch 
import torch.nn as nn 
import torch.nn.functional as F
from molecule_encoder import fp_encoder
import numpy as np 
import gpytorch
from torch.distributions import MultivariateNormal

class ensemble_deep_gp(nn.Module):
    def __init__(self, args):
        super(ensemble_deep_gp, self).__init__()
        self.args = args
        self.kernel_list = nn.ModuleList()
        self.encoder_list = nn.ModuleList()
        num_encoder = args.num_encoder
        for _ in range(num_encoder):
            if 'RGB' in self.args.encode_method:
                self.kernel_list.append(gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()))
            else:
                self.kernel_list.append(gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5)))
            if args.dataset == 'fsmol':
                if 'augment' in args.encode_method:
                    self.encoder_list.append(fp_encoder.MLP_encoder(2215, 1000, 1000, 5))
                else:
                    self.encoder_list.append(fp_encoder.MLP_encoder(2048, 1000, 1000, 5))
            elif args.dataset == 'pQSAR':
                self.encoder_list.append(fp_encoder.MLP_encoder(1024, 500, 500, 5))

        self.num_encoder = num_encoder
    def kernel_parameters(self):
        return [param for kernel in self.kernel_list for param in kernel.parameters()]
    def mlp_parameters(self):
        return [param for encoder in self.encoder_list for param in encoder.parameters()]
    def forward(self, task_data, device):
        x_support, y_support, x_query, y_query = task_data
        processed_x, processed_y, support_num, query_num  = self.get_method_specific_data_format(x_support, y_support, x_query, y_query, device)
        total_neg_likelihood = 0
        num_data = support_num + query_num
        encoder_dic = dict()
        pred_ls = []
        pred_mse_loss = 0
        penalty = 0
        cov_mat_ls = []
        
        if self.args.allow_NCL:
            for idx in range(self.num_encoder):
                encoded_x = self.encoder_list[idx](processed_x)
                encoder_dic[idx] = encoded_x
                log_likelihood, cov_temp = self.multivariate_gaussian_loglikelihood(encoded_x, processed_y, idx, device)
                total_neg_likelihood = total_neg_likelihood + log_likelihood
                pred_temp = self.pred(encoded_x[:support_num], processed_y[:support_num], encoded_x[support_num:], idx, device)
                pred_ls.append(pred_temp)
            mean_output = torch.stack(pred_ls).mean(dim=0)
            beta = 0.001
            for i in range(self.num_encoder):
                pred_mse_loss += torch.mean(0.5 * (pred_ls[i] - processed_y[support_num:]) ** 2)
                penalty_temp = 0
                for j in range(self.num_encoder):
                    if i != j:
                        penalty_temp += (pred_ls[j] - mean_output)
                penalty += beta*torch.mean((pred_ls[i] - mean_output) * penalty_temp)
            return (-total_neg_likelihood)/(num_data) + pred_mse_loss + penalty 
        else:
            for idx in range(self.num_encoder):
                encoded_x = self.encoder_list[idx](processed_x)
                encoder_dic[idx] = encoded_x
                log_likelihood, cov_temp = self.multivariate_gaussian_loglikelihood(encoded_x, processed_y, idx, device)
                total_neg_likelihood = total_neg_likelihood + log_likelihood
            return (-total_neg_likelihood)/(num_data)

    def prediction(self, task_data, device):
        x_support, y_support, x_query, y_query = task_data
        processed_x, processed_y, support_num, query_num = self.get_method_specific_data_format(x_support, y_support, x_query, y_query, device)
        pred = 0
        for idx in range(self.num_encoder):
            encoded_x = self.encoder_list[idx](processed_x)
            pred = pred + self.pred(encoded_x[:support_num], processed_y[:support_num], encoded_x[support_num:], idx, device)
        return pred
    
    def prediction_seperate(self, task_data, device):
        x_support, y_support, x_query, y_query = task_data
        processed_x, processed_y, support_num, query_num = self.get_method_specific_data_format(x_support, y_support, x_query, y_query, device)
        pred = []
        for idx in range(self.num_encoder):
            encoded_x = self.encoder_list[idx](processed_x)
            pred.append(self.pred(encoded_x[:support_num], processed_y[:support_num], encoded_x[support_num:], idx, device))
        return pred
    
    def prediction_seperate_var(self, task_data, device):
        x_support, y_support, x_query, y_query = task_data
        processed_x, processed_y, support_num, query_num = self.get_method_specific_data_format(x_support, y_support, x_query, y_query, device)
        pred_var = []
        for idx in range(self.num_encoder):
            encoded_x = self.encoder_list[idx](processed_x)
            pred_var.append(self.pred_var(encoded_x[:support_num], processed_y[:support_num], encoded_x[support_num:], idx, device))
        return pred_var
    
    def get_method_specific_data_format(self, x_support, y_support, x_query, y_query, device):
        processed_y = torch.cat((y_support, y_query), dim = 0).to(device)
        processed_x = torch.cat((x_support, x_query), dim = 0).to(device)
        num_data_support = x_support.shape[0]
        num_data_query = x_query.shape[0]
        return processed_x, processed_y, num_data_support, num_data_query

    def pred(self, x_support_encoded, y_support, x_query_encoded, kernel_idx, device):
        out = torch.cat((x_support_encoded, x_query_encoded), dim = 0)
        K_all = self.kernel_list[kernel_idx](out).to_dense()
        K_support = K_all[:x_support_encoded.shape[0], :x_support_encoded.shape[0]]
        K_query = K_all[x_support_encoded.shape[0]:, :x_support_encoded.shape[0]]
        mat_inverse = torch.inverse(K_support + 0.01 * torch.eye(K_support.shape[0]).to(device))
        mean = torch.matmul(torch.matmul(K_query, mat_inverse), y_support.to(device))
        return mean
    
    def pred_var(self, x_support_encoded, y_support, x_query_encoded, kernel_idx, device):
        out = torch.cat((x_support_encoded, x_query_encoded), dim=0)
        K_all = self.kernel_list[kernel_idx](out).to_dense()
        K_support = K_all[:x_support_encoded.shape[0], :x_support_encoded.shape[0]]
        K_query = K_all[x_support_encoded.shape[0]:, :x_support_encoded.shape[0]]
        K_query_query = K_all[x_support_encoded.shape[0]:, x_support_encoded.shape[0]:]
        
        mat_inverse = torch.inverse(K_support + 0.01 * torch.eye(K_support.shape[0]).to(device))
        var = torch.diag(K_query_query - torch.matmul(torch.matmul(K_query, mat_inverse), K_query.t()))
        return var
    
    def multivariate_gaussian_loglikelihood(self, x, y, kernel_idx, device):
        covariance = self.kernel_list[kernel_idx](x).to_dense() + 0.01 * torch.eye(x.shape[0]).to(device)
        # Create a MultivariateNormal distribution
        dist = MultivariateNormal(loc=torch.zeros(y.shape[-1]).to(device), covariance_matrix=covariance)
        # Calculate the log-likelihood
        log_likelihood = dist.log_prob(y)
        return log_likelihood, covariance
    
    # def multivariate_gaussian_loglikelihood(self, x, y, kernel_idx, device):
    #     covariance = self.kernel_list[kernel_idx](x).to_dense() + 0.01 * torch.eye(x.shape[0]).to(device)
    #     # K = self.kernel(x).to_dense()
    #     neg_log_likelihood = -self.data_fit_loss(covariance, y, device) - self.penalty(covariance, device) + y.shape[0] * 0.5 * torch.log(torch.tensor(2 * np.pi))
    #     return -neg_log_likelihood, covariance
    
    # def data_fit_loss(self, K, y, device):
    #     inverse_mat = torch.inverse(K + 0.01 * torch.eye(K.shape[0]).to(device))
    #     log_likelihood = -0.5 * torch.matmul(y.T, torch.matmul(inverse_mat, y))
    #     return log_likelihood
    
    # def penalty(self, K, device):
    #      covar_mat = K + 0.01 * torch.eye(K.shape[0]).to(device)
    #      penalty = -0.5 * torch.log(torch.norm(covar_mat))
    #      return penalty