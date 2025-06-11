import torch

# Assume self.extra_params is a tensor with some initial values
class TestClass:
    def __init__(self):
        self.extra_params = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), requires_grad=True)

        # merged_policies and replaced_policy indices
        merged_policies = [0, 1]
        replaced_policy = 2

        # Calculate the average of the extra_params of the two merged policies
        merged_params = (self.extra_params[merged_policies[0]] + self.extra_params[merged_policies[1]]) / 2

        # Detach the tensor from the computation graph and clone it
        extra_params_copy = self.extra_params.detach().clone()

        # Replace the specific row with the merged_params
        extra_params_copy[replaced_policy] = merged_params

        # Reassign the modified tensor back to self.extra_params as an nn.Parameter with requires_grad=True
        self.extra_params = torch.nn.Parameter(extra_params_copy, requires_grad=True)

        # Print the new parameters for debugging
        print('new params of replaced_policy:', self.extra_params[replaced_policy])
        print('params of merged_policies[0]:', self.extra_params[merged_policies[0]])
        print('params of merged_policies[1]:', self.extra_params[merged_policies[1]])

        # Change self.extra_params[merged_policies[0]]
        self.extra_params.data[merged_policies[0]] = torch.tensor([7.0, 8.0])
        # self.extra_params.data[replaced_policy] = torch.tensor([9.0, 10.0])

        # Print the parameters again to see if replaced_policy changed
        print('new params of replaced_policy after change:', self.extra_params[replaced_policy])
        print('params of merged_policies[0] after change:', self.extra_params[merged_policies[0]])
        print('params of merged_policies[1] after change:', self.extra_params[merged_policies[1]])


if __name__ == '__main__':
    test = TestClass()