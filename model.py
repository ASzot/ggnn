import torch
import torch.nn as nn

class ListModule(object):
    """
    In PyTorch it does not currently work to have an array of variables in a
    module. These parameters will not be included in the overall module's
    parameter list. To get around this you need manually add them to the
    module. This is then a helper to access the particular sub-module from the
    "list"

    See https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    for more disucssion.
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))



class PropModel(nn.Module):
    def __init__(self, state_dim, n_nodes, n_edge_types):
        super().__init__()

        self.n_nodes = n_nodes
        self.n_edge_types = n_edge_types
        self.reset_gate = nn.Sequential(
                nn.Linear(state_dim * 3, state_dim),
                nn.Sigmoid()
            )

        self.update_gate = nn.Sequential(
                nn.Linear(state_dim * 3, state_dim),
                nn.Sigmoid()
            )

        self.transform = nn.Sequential(
                nn.Linear(state_dim * 3, state_dim),
                nn.Tanh()
            )

    def forward(self, state_in, state_out, state_cur, adj_matrix):
        A_in = adj_matrix[:, :, :self.n_nodes * self.n_edge_types]
        A_out = adj_matrix[:, :, self.n_nodes * self.n_edge_types:]

        # bmm is a Batch matrix multiplication
        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)

        # Equation 2 of section 3.2 in the paper
        # Which nodes can pass information to other nodes between edges.
        a = torch.cat((a_in, a_out, state_cur), 2)

        # Equation 3 of section 3.2
        z = self.update_gate(a)

        # Equation 4 of section 3.2
        r = self.reset_gate(a)

        # Equation 5  of section 3.2
        h_hat = self.transform(torch.cat((a_in, a_out, r * state_cur), 2))

        # Equation 6 of section 3.2
        output = (1 - z) * state_cur + z * h_hat

        return output


class GGNN(nn.Module):
    def __init__(self, state_dim, annotation_dim, n_edge_types, n_nodes,
            n_steps):
        super().__init__()

        self.state_dim = state_dim
        self.annotation_dim = annotation_dim
        self.n_edge_types = n_edge_types
        self.n_nodes = n_nodes
        self.n_steps = n_steps

        for i in range(self.n_edge_types):
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)

            self.add_module('in_%i' % i, in_fc)
            self.add_module('out_%i' % i, out_fc)

        self.prop_model = PropModel(self.state_dim, self.n_nodes,
                self.n_edge_types)

        self.out = nn.Sequential(
                nn.Linear(self.state_dim + self.annotation_dim,
                    self.state_dim),
                nn.Tanh(),
                nn.Linear(self.state_dim, 1)
            )

        # Initialize nodes
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

        self.in_fcs = AttrProxy(self, 'in_')
        self.out_fcs = AttrProxy(self, 'out_')


    def forward(self, init_hidden_state, annotation, adj_matrix):
        hidden_state = init_hidden_state

        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_fc = self.in_fcs[i]
                out_fc = self.out_fcs[i]

                # Apply equation 8 from section 4 to infer the intermediate
                # annotations.
                in_states.append(in_fc(hidden_state))
                out_states.append(out_fc(hidden_state))

            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_nodes * self.n_edge_types,
                    self.state_dim)

            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_nodes * self.n_edge_types,
                    self.state_dim)

            hidden_state = self.prop_model(in_states, out_states, hidden_state, adj_matrix)

        # Equation 7 of section 3.3
        output = self.out(torch.cat((hidden_state, annotation), 2))
        output = output.sum(2)

        return output


