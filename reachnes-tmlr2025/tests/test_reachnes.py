# import networkx as nx
# import torch
#
# import reachnes.optim
#
# import reachnes.loss
# import reachnes.reachnes as rn
# import reachnes.reduction as rn_reduce
# import reachnes.coeffs as rn_coeffs
#
#
# def main():
#     torch.autograd.set_detect_anomaly(True)
#
#     cycle = nx.cycle_graph(100, create_using=nx.DiGraph)
#     adj = nx.adjacency_matrix(cycle).T
#     num_series = 1
#     param = rn.RNParams(directed=True, normalization_seq="O", dtype=torch.float, memory_available=8,
#                         thresholding=True, threshold2sparse=False)
#     # reduction_model = rn_reduce.SPRSVDProximalReduction(num_nodes=100,  k=8, num_series=num_series, num_oversampling=5,
#     #                                                     block_size=2)
#     reduction_model = rn_reduce.ECFReduction(num_eval_points=8, num_series=num_series, learnable_max_val=True)
#     # reduction_model = rn_reduce.SVDProximalReduction(emb_dim=8, include_v=True)
#     # coeffs_params = rn_coeffs.CoeffsParameters(num_series=num_series, order=5, learnable=True, normalize=True,
#     #                                            initialization='geometric', init_param_min=0.8, init_param_max=0.99)
#     coeffs_params = rn_coeffs.CoeffsParameters(num_series=num_series, order=5, learnable=True, normalize=True,
#                                                initialization='poisson', init_param_min=1, init_param_max=1)
#
#     model = rn.Reachnes(adj=adj,
#                         params=param,
#                         reduction_model=reduction_model,
#                         coeffs_params=coeffs_params
#                         )
#     device = torch.device('cpu')
#     model = model.custom_to(device)
#
#     print(model.adj_obj.device)
#     print(model.coeffs_obj.log_coeffs_.device)
#
#     embeddings, node_indices = model(return_series=True)
#
#     loss = reachnes.loss.push_loss(embeddings, num_batches=10)
#     loss.backward()
#
#     print(embeddings.shape)
#     print(node_indices)
#     print(model.coeffs_obj.log_coeffs_.grad)
#     print(reduction_model.log_max_val.grad)
#     print(loss.item())
#
#
# if __name__ == "__main__":
#     main()
