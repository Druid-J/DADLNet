import xlsxwriter as xw


def writerxlsx_1(path, data, dataname):
    workbook = xw.Workbook(path)
    worksheet = workbook.add_worksheet(dataname)
    row = 0
    col = 0
    for fold, loss, acc, tp, fp, tn, fn, recall, speci, precision,f1 in data:
        worksheet.write(row, col, fold)
        worksheet.write(row, col+1, loss)
        worksheet.write(row, col + 2, acc)
        worksheet.write(row, col + 3, tp)
        worksheet.write(row, col + 4, fp)
        worksheet.write(row, col + 5, tn)
        worksheet.write(row, col + 6, fn)
        worksheet.write(row, col + 7, recall)
        worksheet.write(row, col + 8, speci)
        worksheet.write(row, col + 9, precision)
        worksheet.write(row, col + 10, f1)
        row += 1
    workbook.close()


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss