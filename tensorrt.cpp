ILayer* msgConvBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int s, std::vector<int> kernels, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int min_ch = c2 / 2;
    int groups = kernels.size();

    // First Conv Layer
    auto cv1 = convBlock(network, weightMap, input, min_ch, 1, s, 1, lname + ".cv1");
    assert(cv1);

    std::vector<ITensor*> stack_tensors;
    // Loop through each kernel size
    for (int i = 0; i < kernels.size(); ++i) {
        auto gconv = convBlock(network, weightMap, *cv1->getOutput(0), min_ch / 2, kernels[i], 1, min_ch / 2, lname + ".convs." + std::to_string(i));
        assert(gconv);
        stack_tensors.push_back(gconv->getOutput(0));
    }

    // Concatenate along the channel dimension
    auto cat = network->addConcatenation(stack_tensors.data(), stack_tensors.size());
    assert(cat);
    cat->setAxis(1);  // Channel dimension

    // Final 1x1 Conv Layer
    auto conv1x1 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".conv1x1");
    assert(conv1x1);

    return conv1x1;
}

ILayer* msgaConvBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int s, std::vector<int> kernels, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int min_ch = c2 / 2;
    int groups = kernels.size();

    // Initial Conv Layer based on stride
    ILayer* cv1;
    if (s == 1) {
        cv1 = convBlock(network, weightMap, input, min_ch, 1, 1, 1, lname + ".cv1");
    } else if (s == 2) {
        cv1 = convBlock(network, weightMap, input, min_ch, 3, 2, 1, lname + ".cv1");
    }
    assert(cv1);

    // Multi-scale Conv Blocks
    std::vector<ITensor*> tensors_to_stack;
    for (int i = 0; i < groups; ++i) {
        // Custom slicing or rearranging may be needed here.
        // Apply Conv Block
        auto gconv = convBlock(network, weightMap, *cv1->getOutput(0), min_ch / 2, kernels[i], 1, min_ch / 2, lname + ".convs." + std::to_string(i));
        assert(gconv);
        tensors_to_stack.push_back(gconv->getOutput(0));
    }

    // Concatenation
    auto cat = network->addConcatenation(tensors_to_stack.data(), tensors_to_stack.size());
    assert(cat);
    cat->setAxis(1);

    // Final 1x1 Conv Layer
    auto conv1x1 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".conv1x1");
    assert(conv1x1);

    // Shortcut
    ILayer* shortcut;
    if (c1 != c2) {
        auto dwconv = dwConvBlock(network, weightMap, input, c1, c1, 3, s, lname + ".shortcut.0");
        assert(dwconv);
        shortcut = convBlock(network, weightMap, *dwconv->getOutput(0), c2, 1, 1, 1, lname + ".shortcut.1");
    } else {
        shortcut = &input;
    }
    assert(shortcut);

    // Elementwise addition
    auto ew = network->addElementWise(*conv1x1->getOutput(0), *shortcut->getOutput(0), ElementWiseOperation::kSUM);
    assert(ew);

    return ew;
}

ILayer* msgaBottleneckBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int k, int s, std::string lname) {
    // Intermediate channels
    int c_ = c2 / 2;

    // Two MSGAConv layers
    auto msgaconv1 = msgaConvBlock(network, weightMap, input, c1, c_, 1, {3, 5}, lname + ".conv.0");
    assert(msgaconv1);
    
    auto msgaconv2 = msgaConvBlock(network, weightMap, *msgaconv1->getOutput(0), c_, c2, 1, {3, 5}, lname + ".conv.1");
    assert(msgaconv2);

    // Shortcut
    ILayer* shortcut;
    if (c1 != c2) {
        auto dwconv = dwConvBlock(network, weightMap, input, c1, c1, k, s, lname + ".shortcut.0");
        assert(dwconv);
        shortcut = convBlock(network, weightMap, *dwconv->getOutput(0), c2, 1, 1, 1, lname + ".shortcut.1");
    } else {
        shortcut = &input;
    }
    assert(shortcut);

    // Elementwise addition
    auto ew = network->addElementWise(*msgaconv2->getOutput(0), *shortcut->getOutput(0), ElementWiseOperation::kSUM);
    assert(ew);

    return ew;
}

ILayer* C3GAhostMSGBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname) {
    // Hidden channels
    int c_ = static_cast<int>(c2 * e);

    // Conv Layers
    auto cv1 = convBlock(network, weightMap, input, c1, c_, 1, 1, 1, lname + ".cv1");
    assert(cv1);

    auto cv2 = convBlock(network, weightMap, input, c1, c_, 1, 1, 1, lname + ".cv2");
    assert(cv2);

    // MSGABottleneck sequence
    ITensor* m_output = cv1->getOutput(0);
    for (int i = 0; i < n; ++i) {
        auto m = msgaBottleneckBlock(network, weightMap, *m_output, c_, c_, 3, 1, lname + ".m" + std::to_string(i));
        assert(m);
        m_output = m->getOutput(0);
    }

    // Concatenate
    ITensor* concatTensors[] = { m_output, cv2->getOutput(0) };
    auto catLayer = network->addConcatenation(concatTensors, 2);
    assert(catLayer);
    catLayer->setAxis(1); // Concatenation along the channel dimension

    // cv3 layer
    auto cv3 = convBlock(network, weightMap, *catLayer->getOutput(0), 2 * c_, c2, 1, 1, 1, lname + ".cv3");
    assert(cv3);

    return cv3;
}
