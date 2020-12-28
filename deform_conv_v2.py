import torch
from torch import nn


class DeformConv2d(nn.Module):
    # inc表示输入通道数
    # outc 表示输出通道数
    # kernel_size表示卷积核尺寸
    # stride 卷积核滑动步长
    # bias 偏置
    # modulation DCNV1还是DCNV2的开关
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        # 普通的卷积层，即获得了偏移量之后的特征图再接一个普通卷积
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        # 获得偏移量，卷积核的通道数应该为2xkernel_sizexkernel_size
        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        # 偏移量初始化为0
        nn.init.constant_(self.p_conv.weight, 0)
        # 注册module反向传播的hook函数, 可以查看当前层参数的梯度
        self.p_conv.register_backward_hook(self._set_lr)
        # 将modulation赋值给当前类
        self.modulation = modulation
        if modulation:
            # 如果是DCN V2，还多了一个权重参数，用m_conv来表示
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            # 注册module反向传播的hook函数, 可以查看当前层参数的梯度
            self.m_conv.register_backward_hook(self._set_lr)

    # 静态方法 类或实例均可调用，这函数的结合hook可以输出你想要的Variable的梯度
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    # 前向传播函数
    def forward(self, x):
        # 获得输入特征图x的偏移量
        # 假设输入特征图shape是[1,3,32,32]，然后卷积核是3x3，
        # 输出通道数为32，那么offset的shape是[1,2*3*3,32]
        offset = self.p_conv(x)
        # 如果是DCN V2那么还需要获得输入特征图x偏移量的权重项
        # 假设输入特征图shape是[1,3,32,32]，然后卷积核是3x3，
        # 输出通道数为32，那么offset的权重shape是[1,3*3,32]
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))
        # dtype = torch.float32
        dtype = offset.data.type()
        # 卷积核尺寸大小
        ks = self.kernel_size
        # N=2*3*3/2=3*3=9
        N = offset.size(1) // 2
        # 如果需要Padding就先Padding
        if self.padding:
            x = self.zero_padding(x)

        # p的shape为(b, 2N, h, w)
        # 这个函数用来获取所有的卷积核偏移之后相对于原始特征图x的坐标（现在是浮点数）
        p = self._get_p(offset, dtype)

        # 我们学习出的量是float类型的，而像素坐标都是整数类型的，
        # 所以我们还要用双线性插值的方法去推算相应的值
        # 维度转换，现在p的维度为(b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        # floor是向下取整
        q_lt = p.detach().floor()
        # +1相当于原始坐标向上取整
        q_rb = q_lt + 1
        # 将q_lt即左上角坐标的值限制在图像范围内
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        # 将q_rb即右下角坐标的值限制在图像范围内
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        # 用q_lt的前半部分坐标q_lt_x和q_rb的后半部分q_rb_y组合成q_lb
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        # 同理
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # 对p的坐标也要限制在图像范围内
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        # 双线性插值的4个系数
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        # 现在只获取了坐标值，我们最终木的是获取相应坐标上的值，
        # 这里我们通过self._get_x_q()获取相应值。
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        # 双线性插值计算
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        # 在获取所有值后我们计算出x_offset，但是x_offset的size
        # 是(b,c,h,w,N)，我们的目的是将最终的输出结果的size变
        # 成和x一致即(b,c,h,w)，所以在最后用了一个reshape的操作。
        # 这里ks=3
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out
    
    # 通过函数_get_p_n生成了卷积的相对坐标，其中卷积的中心点被看成原点
    # 然后其它点的坐标都是相对于原点来说的，例如self.kernel_size=3，通
    # 过torch.meshgrid生成从（-1，-1）到（1，1）9个坐标。将坐标的x和y
    # 分别存储，然后再将x，y以(1,2N,1,1)的形式返回，这样我们就获取了一
    # 个卷积核的所有相对坐标。
    def _get_p_n(self, N, dtype):
        # p_n_x = tensor([[-1, -1, -1],
        # [ 0,  0,  0],
        # [ 1,  1,  1]])
        # p_n_y = tensor([[-1,  0,  1],
        # [-1,  0,  1],
        # [-1,  0,  1]])
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        # p_n = tensor([-1, -1, -1,  0,  0,  0,  1,  1,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1])
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        # p_n.shape=[1,2*N,1,1]
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    # 获取卷积核在特征图上对应的中心坐标，也即论文公式中的p_0
    # 通过torch.mershgrid生成所有的中心坐标，然后通过kernel_size
    # 推断初始坐标，然后通过stride推断所有的中心坐标，这里注意一下，
    # 代码默认torch.arange从1开始，实际上这是kernel_size为3时的情况，
    # 严谨一点torch.arrange应该从kernel_size//2开始，这个实现只适合3x3的卷积。
    def _get_p_0(self, h, w, N, dtype):
        # 设w = 7, h = 5, stride = 1
        # 有p_0_x = tensor([[1, 1, 1, 1, 1, 1, 1],
        # [2, 2, 2, 2, 2, 2, 2],
        # [3, 3, 3, 3, 3, 3, 3],
        # [4, 4, 4, 4, 4, 4, 4],
        # [5, 5, 5, 5, 5, 5, 5]])
        # p_0_x.shape = [5, 7]
        # p_0_y = tensor([[1, 2, 3, 4, 5, 6, 7],
        # [1, 2, 3, 4, 5, 6, 7],
        # [1, 2, 3, 4, 5, 6, 7],
        # [1, 2, 3, 4, 5, 6, 7],
        # [1, 2, 3, 4, 5, 6, 7]])
        # p_0_y.shape = [5, 7]
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        # p_0_x的shape为torch.Size([1, 9, 5, 7])
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        # p_0_y的shape为torch.Size([1, 9, 5, 7])
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        # p_0的shape为torch.Size([1, 18, 5, 7])
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        # N = 18 / 2 = 9
        # h = 32
        # w = 32
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        # 卷积坐标加上之前学习出的offset后就是论文提出的公式(2)也就是加上了偏置后的卷积操作。
        # 比如p(在N=0时)p_0就是中心坐标，而p_n=(-1,-1)，所以此时的p就是卷积核中心坐标加上
        # (-1,-1)(即红色块左上方的块)再加上offset。同理可得N=1,N=2...分别代表了一个卷积核
        # 上各个元素。
        p = p_0 + p_n + offset
        return p

    # 通过self._get_x_q()获取偏移坐标对应的值
    # 
    def _get_x_q(self, x, q, N):
        # 输入x是我们最早输入的数据x，q则是我们的坐标信息。
        # 首先我们获取q的相关尺寸信息(b,h,w,2N)，再获取x
        # 的w保存在padding_w中，将x(b,c,h,w)通过view变成(b,c,h*w)。
        # 这样子就把x的坐标信息压缩在了最后一个维度(h*w)，这样做的
        # 目的是为了使用tensor.gather()通过坐标来获取相应值。
        # (这里注意下q的h,w和x的h,w不一定相同，比如stride不为1的时候)
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)
        # 同样地，由于(h,w)被压缩成了(h*w)所以在这个维度上，每过w个
        # 元素，就代表了一行，所以我们的坐标index=offset_x*w+offset_y
        # (这样就能在h*w上找到(h,w)的相应坐标)同时再把偏移expand()到
        # 每一个通道最后返回x_offset(b,c,h,w,N)。(最后输出x_offset的
        # h,w指的是x的h,w而不是q的)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        # 函数首先获取了x_offset的所有size信息，然后以kernel_size为
        # 单位进行reshape，因为N=kernel_size*kernel_size，所以我们
        # 分两次进行reshape，第一次先把输入view成(b,c,h,ks*w,ks)，
        # 第二次再view将size变成(b,c,h*ks,w*ks)

        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset
