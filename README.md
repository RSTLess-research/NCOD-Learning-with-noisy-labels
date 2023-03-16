
# Loss Function for Improved Learning with Noisy Labels

This repository contains an implementation of the loss function proposed in the paper "Combining Distance to Class Centroids and Outlier Discounting for Improved Learning with Noisy Labels". The loss function can be used to train  deep neural networks especially in the presence of noisy labels.

## Usage

To use the loss function, follow these instructions:

### Step 1:

Declare the following variables outside the class:

- `mean`: The initialization mean for parameter  `u`
- `std`: The initialization standard deviation for parameter  `u`
- `encoder_features`: Number of features `ϕ(xi)` the output of the second to the last layer of our network
- `total_epochs`: The total number of epochs the network is trained for

It is preferable to pass the above four variables using the config file. For simplicity, they have been declared global in this implementation.

### Step 2:

Initialize the `ncodLoss` class with the following parameters:

- `sample_labels`: The list of training labels for all samples "should not be one hot encoded"
- `num_examp`: Total number of training samples
- `num_classes`: Total number of training classes

If you want to use the two additional regularization terms "the consistency regularizer LC and class-balance regularizer LB", pass the following parameters:

- `ratio_consistency` The weightage given to `LC`. Check our paper for its value. Default is zero.
- `ratio_balance` he weightage given to `LB`. Check our paper for its value. Default is zero.

Additional information for step 2:

- `self.beginning`: It is turned to `True` because we start saving the average class latent representation from the beginning. Turn this to `False` at the time of saving the average latent representation of the class so that it will be turned on again by the function after the first run.

### Step 3:

Call the `forward` method of `ncodLoss` with the following parameters:

- `index`: The index of each training sample in the current batch.
- `outputs`: The target output generated by the model (M).
- `labels`: one-hot encoded representation of each sample in the batch.
- `out`: encoded feature representation of each sample in a batch.
- `flag`:  The batch number or batch_id for each batch.
- `epoch`: The number of the current epoch.

Note that in the case of the two-networked ensembles architecture, the size of `outputs` and `out` will be twice that of the number of indices because the second chunk contains the result of the augmented data of the first chunk.

### Step 4:

Create an object of the above class `ncodLoss` and get the weights of `u`. Assign some learning rate to `u` (for learning rate check the paper) with zero weight decay and create a separate optimizer for `u`. We used SGD. Let us call it `optimizer_u`.

### Step 5:

During training of the network, optimize the weights of `u` as well. For example:

      for current_epoch in total_number_of_epochs:
      
          for batch_number, (Actualdata, Augumented, label, indexs) in yourdataloader:
      
              Actualdata, label = Actualdata.to(device), label.long().to(device)
      
              target = torch.zeros(len(label),your_number_of_classes).to(device).scatter_(1, label.view(-1,1), 1)
      
              if (you want to use the additional regularisation of LC and LB) > 0:
                  Augumented = Augumented.to(self.device)
                  data = torch.cat([Actualdata, Augumented]).cuda()
              else:
                  data = Actualdata
      
              output,out = your_model(data)
      
              loss = object_of_ncodLoss(indexs, output, target, out, batch_number, epoch)
      
              self.optimizer_u.zero_grad()
              self.your_own_optimizer.zero_grad()
      
              loss.backward()
      
              self.optimizer_u.step()
              self.your_own_optimizer.step()



















