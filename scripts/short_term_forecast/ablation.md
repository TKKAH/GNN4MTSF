# Comparative Experiment
First,cd to the Short_term_forecast dir:
```
cd Short_term_forecast
```
## Myen
```
 python test.py --config_filename=data/model/para_bay.yaml --temperature=0.5
```
```
 python test.py --config_filename=data/model/para_la.yaml --temperature=0.5
```

## without AGL
1. Comment the code in ```model/pytorch/model.py``` and add the code next:```adj=torch.ones(self.num_nodes, self.num_nodes).to(device)```
    ```
        adj = gumbel_softmax(x, temperature=temp, hard=True)
        adj = adj[:, 0].clone().reshape(self.num_nodes, -1)
        mask = torch.eye(self.num_nodes, self.num_nodes).bool().to(device)
        # 将被遮蔽位置上的元素置为 0
        adj.masked_fill_(mask, 0)
    ```
2. Execute the command:
    ```
    python test.py --config_filename=data/model/without_agl_para_bay.yaml --temperature=0.5
    ```
    ```
    python test.py --config_filename=data/model/without_agl_para_la.yaml --temperature=0.5
    ```
## without GCN
1. Set the param ```use_gc_for_ru``` of the DCGRUCell class init function in the ```model/pytorch/cell.py``` False, comment the code in the same class forward function.

    Then add the code next:```c=fn(inputs, adj_mx, r * hx, self._num_units)```
    ```
        c = self._gconv(inputs, adj_mx, r * hx, self._num_units)
    ```
2. Execute the command:
    ```
    python test.py --config_filename=data/model/without_gcn_para_bay.yaml --temperature=0.5
    ```
    ```
    python test.py --config_filename=data/model/without_gcn_para_la.yaml --temperature=0.5
    ```