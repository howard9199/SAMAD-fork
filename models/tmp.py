        overall_loss = None
        #loss_fct = nn.CrossEntropyLoss()
        loss_fct = self.kappa_loss
        print(output.shape)
        print(labels.shape)
        if labels is not None:
            labels = labels.long().to(self.device)
            loss_holistic = loss_fct(output.view(-1, self.num_labels), labels.view(-1))
            overall_loss = loss_holistic
            overall_loss = overall_loss.to('cuda')
        # soft label
        # if labels is not None:
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss_holistic = loss_fct(output.view(-1, self.num_labels), labels)
        #     overall_loss = loss_holistic