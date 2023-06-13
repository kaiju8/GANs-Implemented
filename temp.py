num_centers = len(classes)
weights = torch.ones(num_centers,requires_grad=True)
means = torch.tensor(np.random.randn(len(classes),size*size),requires_grad=True)
stdevs = torch.tensor(np.abs(np.random.randn(len(classes),size*size)),requires_grad=True)

parameters = [weights, means, stdevs]
optimizer1 = optim.SGD(parameters, lr=0.001, momentum=0.9)

num_iter = 10
for i in range(num_iter):
    mix = D.Categorical(weights)
    comp = D.Independent(D.Normal(means,stdevs), 1)
    gmm = D.MixtureSameFamily(mix, comp)

    optimizer1.zero_grad()
    x = images_tensor #this can be an arbitrary x samples
    loss2 = -gmm.log_prob(x).mean()#-densityflow.log_prob(inputs=x).mean()
    loss2.backward()
    optimizer1.step()

    print(i, loss2)




for epoch in range(epochs):
    for i, data in enumerate(train_loader):
        images, labels = data
        mix = D.Categorical(weights)
        comp = D.Independent(D.Normal(means,stdevs), 1)
        gmm = D.MixtureSameFamily(mix, comp)

        optimizer1.zero_grad()
        x = images #this can be an arbitrary x samples
        loss2 = -gmm.log_prob(x).mean()#-densityflow.log_prob(inputs=x).mean()
        loss2.backward()
        optimizer1.step()

        print(i, loss2)

