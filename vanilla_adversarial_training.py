import os
import numpy as np
import torch

class Attack:
    def __init__(self, num_iter, lr, targeted):
        self.num_iter = num_iter
        self.lr = lr
        self.cls_criterion = torch.nn.CrossEntropyLoss()
        self.targeted = targeted

class FGSM(Attack):
    def __init__(self, num_iter, lr, targeted):
        super(FGSM, self).__init__(num_iter, lr, targeted)
        self.num_iter = 1

    def __call__(self, model, data, target_label):
        data.requires_grad_()
        raw_prediction = model(data)
        if not self.targeted:
            target_label = torch.argmax(raw_prediction, -1)
        cls_loss = self.cls_criterion(raw_prediction, target_label)
        cls_loss.backward()
        grad = data.grad
        data = data.detach()
        if self.targeted:
            data -= grad.sign() * self.lr
        else:
            data += grad.sign() * self.lr
        data[data > 1] = 1
        data[data < 0] = 0
        return data

class PGD(Attack):
    def __init__(self, num_iter, lr, targeted):
        super(PGD, self).__init__(num_iter, lr, targeted)

    def __call__(self, model, data, target_label):
        raw_prediction = model(data)
        if not self.targeted:
            target_label = torch.argmax(raw_prediction, -1)
        for i in range(self.num_iter):
            data.requires_grad_()
            raw_prediction = model(data)
            cls_loss = self.cls_criterion(raw_prediction, target_label)
            cls_loss.backward()
            grad = data.grad
            data = data.detach()
            if self.targeted:
                data -= grad * self.lr
            else:
                data += grad * self.lr
            data[data > 1] = 1
            data[data < 0] = 0
        return data

class CW(Attack):
    def __init__(self, num_iter, lr, targeted):
        super(CW, self).__init__(num_iter, lr, targeted)

    def __call__(self, model, data, target_label):
        raw_prediction = model(data)
        if not self.targeted:
            target_label = torch.argmax(raw_prediction, -1)
        for i in range(self.num_iter):
            data.requires_grad_()
            raw_prediction = model(data)
            tmp = raw_prediction[0].clone().detach()
            largest = torch.argmax(tmp)
            tmp[largest] = -1e9
            second_largest = torch.argmax(tmp)
            cls_loss = raw_prediction[0, largest] - raw_prediction[0, second_largest]
            cls_loss.backward()
            grad = data.grad
            data = data.detach()
            if self.targeted:
                data -= grad * self.lr
            else:
                data += grad * self.lr
            data[data > 1] = 1
            data[data < 0] = 0
        return data

class VAT:
    """
    adopt_details: A bool
    num_iter_attack: A int
    lr_attack: A float
    targeted_attack: A bool
    A: number of attack methods (currently 3)
    """
    def __init__(self, model, adopt_details, num_iter_attack, lr_attack, targeted_attack):
        self.methods = ['fgsm', 'pgd', 'cw']
        assert len(adopt_details) == len(self.methods), \
            'adoption details not compatible with available attack methods'
        assert len(num_iter_attack) == len(self.methods), \
            'numbers of adversarial iterations not compatible with available attack methods'
        assert len(lr_attack) == len(self.methods), \
            'attack step lengths not compatible with available attack methods'
        assert len(targeted_attack) == len(self.methods), \
            'targeted details not compatible with available attack methods'
        self.attack = {
            'fgsm': {'adopt': adopt_details[0], 'attacker': FGSM(num_iter_attack[0], lr_attack[0], targeted_attack[0])},
            'pgd': {'adopt': adopt_details[1], 'attacker': PGD(num_iter_attack[1], lr_attack[1], targeted_attack[1])},
            'cw': {'adopt': adopt_details[2], 'attacker': CW(num_iter_attack[2], lr_attack[2], targeted_attack[2])},
        }
        self.adopted_methods = np.array(list(self.attack.keys()))[np.array(adopt_details)]
        self.model = model

    def data(self, data, target_label):
        """
        data: NCHW torch.Tensor
        target_label: N torch.Tensor
        """
        self.model.eval()
        adv_data = data.clone()
        batch_size = data.size(0)
        slice_len = batch_size // len(self.adopted_methods)
        for i in range(len(self.adopted_methods)):
            data_slice = data[i * slice_len : (i + 1) * slice_len]
            target_label_slice = target_label[i * slice_len : (i + 1) * slice_len]
            adv_data_slice = self.attack[self.adopted_methods[i]]['attacker'](self.model, data_slice, target_label_slice)
            adv_data[i * slice_len : (i + 1) * slice_len] = adv_data_slice
        self.model.train()
        return adv_data

    def eval(self, data, original_label, target_label):
        """
        data: NCHW torch.Tensor
        original_label: N torch.Tensor
        target_label: N torch.Tensor
        output: [number of samples, clean sample acc, attack 1 acc, attack 2 acc, ...]
        """
        self.model.eval()
        original_label = original_label.detach().cpu().numpy()
        raw_prediction = self.model(data).detach().cpu().numpy()
        prediction_index = np.argmax(raw_prediction, -1)
        num_correct_clean = np.sum((prediction_index == original_label).astype(np.float))
        result = [data.size(0), num_correct_clean]
        for method in self.adopted_methods:
            adv_data = self.attack[method]['attacker'](self.model, data, target_label)
            raw_prediction = self.model(adv_data).detach().cpu().numpy()
            prediction_index = np.argmax(raw_prediction, -1)
            num_correct_adv = np.sum((prediction_index == original_label).astype(np.float))
            result.append(num_correct_adv)
        self.model.train()
        return result

if __name__ == '__main__':
    import torchvision
    model = torchvision.models.vgg16(pretrained=False)
    adv_training_frame = VAT(
        model=model,
        adopt_details=[True, False, True],
        num_iter_attack=[10, 10, 10],
        lr_attack=[1e-2, 1e-5, 1e-5],
        targeted_attack=[True, False, True]
    )
    data = torch.randn(4, 3, 224, 224).float()
    original_label = torch.Tensor([2, 3, 4, 6]).long()
    target_label = torch.Tensor([1, 2, 3, 5]).long()
    adv_data = adv_training_frame.data(data, target_label)
    print(adv_data.size())
    eval_results = adv_training_frame.eval(data, original_label, target_label)
    print(eval_results)

