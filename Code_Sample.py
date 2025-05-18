
#  The code is a demo of
#  ----------------------------------------------

import os
import torch
import torch.nn as nn
import numpy
import joblib
import time
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import torch.distributions as tdist

import warnings

cwd = os.getcwd()

read_path = 'data_a'
os.chdir(read_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Data
data_s = joblib.load('data_s.pkl').to(torch.float32).to(device)  # Data of X
rc_out = joblib.load('rc_out.pkl').to(torch.float32).to(device) # Close return
rl_out = joblib.load('rl_out.pkl').to(torch.float32).to(device) # lowest return

# Training
asset_n = data_s.shape[0] # Asset Numbers
hidden_n = 6
fea_n = data_s.shape[2]
T = 150
bs = 40
s_bar = 0.9 * torch.ones(T, asset_n, device=device)

class my_model(nn.Module):
    def __init__(self, f_n, a_n, h_n):
        super(my_model, self).__init__()
        self.rnn = nn.RNN(f_n, h_n)
        self.lm_x = nn.Linear(a_n * h_n, a_n, bias=True)
        self.lm_s = nn.Linear(a_n * h_n, a_n, bias=True)

    def forward(self, x0):
        x1, _ = self.rnn(x0)
        x2 = x1.view(T, asset_n * hidden_n)
        x3 = self.lm_x(x2)
        x4 = self.lm_s(x2)
        xv = torch.softmax((torch.relu(x3)+0.00001).log(), 1)
        sv = s_bar + (0.9999 - s_bar) * torch.sigmoid(x4)
        return xv, sv


time_b = time.time()

model = my_model(fea_n, asset_n, hidden_n).to(device)
T_all = data_s.shape[1]
T_tra = T_all - T
data_nor = torch.zeros(asset_n, T_all, fea_n, device=device)
for i in range(asset_n):
    for j in range(fea_n):
        ave = data_s[i, 0:T_tra, j].mean()
        sd = data_s[i, 0:T_tra, j].std()
        data_nor[i, :, j] = (data_s[i, :, j] - ave) / sd
model_pa = model.state_dict()
al = 0.005
bs = 40
epock_max = 150
for epoch in range(epock_max):
    rand_n = torch.randperm(T_tra - T)
    gg = 0
    while gg + T < T_tra - T:
        for k in rand_n[gg: gg + bs + 1]:
            fv = 0
            data_e = data_nor[:, k:k + T]
            rc_e = (rc_out[:, k:k + T]).t()
            rl_e = (rl_out[:, k:k + T]).t()
            x, s = model(data_e)
            rev = (s >= rl_e) * s + (s < rl_e) * rc_e
            Re = (x * rev).sum(1).log()

            m_re = Re.mean()
            v_re = Re.std()
            fv1 = m_re / v_re
            fv = fv + fv1
        # fv = fv / bs
        fv.backward()
        for kk in model.parameters():
            kk.data.add_(al * kk.grad)
        model.zero_grad()
        gg = gg + bs

    data_eb = data_nor[:, T_tra:T_all]
    rc_eb = (rc_out[:, T_tra:T_all]).t()
    rl_eb = (rl_out[:, T_tra:T_all]).t()
    x_b, s_b = model(data_eb)
    rev_b = (s_b >= rl_eb) * s_b + (s_b < rl_eb) * rc_eb
    Re_b = (x_b * rev_b).sum(1)
    print(f"epoch: {epoch}, In-of-Sample fv: {fv.item():.4f}, Out-of-Sample CW: {Re_b.prod().item():.4f}")


################################################################
# Results Output

model.eval()
data_eb = data_nor[:, T_tra:T_all]
rc_eb = (rc_out[:, T_tra:T_all]).t().to('cpu')
rl_eb = (rl_out[:, T_tra:T_all]).t().to('cpu')
x_b, s_b = model(data_eb)
x_b = x_b.to('cpu')
s_b = s_b.to('cpu')
rev_b = (s_b >= rl_eb) * s_b + (s_b < rl_eb) * rc_eb

Re_b = (x_b * rev_b).sum(1) ## Return of E2E_With
Re_o = (x_b * rc_eb).sum(1) ## Return of E2E_Without

## Market strategy
R_1n = torch.zeros(T)
ww_1n = torch.ones(asset_n)/asset_n
for tt in range(T):
    R_1n[tt] = (ww_1n * rc_eb[tt]).sum() ## Return of Market
    ww_1n = ww_1n * rc_eb[tt]/R_1n[tt]


# Best-Stock Strategy
t_tt = T_tra
rc_tra = (data_s[:, T_tra-t_tt:T_tra, 0] - 1).to("cpu")
R_si = rc_eb.prod(0)
n_si = R_si.argmax()
Re_si = rc_eb[:, n_si] # Return of 1/N Best-Stock

# Mean-Variance Strategy
rc_mean = rc_tra.mean(1)
rc_cov = torch.zeros(asset_n, asset_n)
for i in range(asset_n):
    for j in range(i, asset_n):
        covv = ((rc_tra[i] - rc_mean[i]) * (rc_tra[j] - rc_mean[j])).sum() / (t_tt-1)
        rc_cov[i, j] = covv
        rc_cov[j, i] = covv

n_m = torch.argmax(rc_mean)
r_max = rc_mean.max()
r_min = rc_mean.min()
rc_mean = rc_mean.numpy()
rc_mean = numpy.float64(rc_mean)
rc_cov = rc_cov.numpy()
rc_cov = numpy.float64(rc_cov)

Gv = numpy.vstack((-rc_mean, numpy.diag((-numpy.ones(asset_n)))))


pp = matrix(rc_cov)
mm = matrix(-rc_mean)
qq = matrix(numpy.zeros(asset_n))
G = matrix(Gv)
A = matrix(numpy.ones(asset_n)).trans()
b = matrix(numpy.ones(1))

r_ran = numpy.linspace(r_min, r_max, 50)

x_mv1 = numpy.zeros((asset_n, 50))
f_mv1 = numpy.zeros(50)
for rr in range(50):
    h = matrix(numpy.concatenate((numpy.array([-r_ran[rr]]), numpy.zeros(asset_n)), axis=0))

    result = solvers.qp(pp, qq, G, h, A, b)

    x_mv1[:, rr] = (numpy.array(result['x'])).squeeze(1)
    f_mv1[rr] = numpy.array(result['primal objective'])

sr = r_ran/numpy.sqrt(2*f_mv1)
n_sr = sr.argmax()
x_vv = x_mv1[:, 0]
x_rr = x_mv1[:, -1]
x_sr = x_mv1[:, n_sr]
x_vv = torch.from_numpy(x_vv)
x_rr = torch.from_numpy(x_rr)
x_sr = torch.from_numpy(x_sr)

Re_vv = (x_vv.squeeze(0)*rc_eb).sum(1)
Re_sr = (x_sr.squeeze(0)*rc_eb).sum(1)
Re_rr = (x_rr.squeeze(0)*rc_eb).sum(1)

Re_b = (Re_b.detach()).numpy()
cw_b = numpy.ones(T+1)
cw_o = numpy.ones(T+1)
cw_si = numpy.ones(T+1)
cw_1n = numpy.ones(T+1)
cw_vv = numpy.ones(T+1)
cw_rr = numpy.ones(T+1)
cw_sr = numpy.ones(T+1)

for tt in range(1, T+1):
    cw_b[tt] = cw_b[tt-1]*(Re_b[tt-1])
    cw_o[tt] = cw_o[tt - 1] * Re_o[tt - 1]
    cw_si[tt] = cw_si[tt - 1] * Re_si[tt - 1]
    cw_1n[tt] = cw_1n[tt - 1] * R_1n[tt - 1]
    cw_vv[tt] = cw_vv[tt - 1] * Re_vv[tt - 1]
    cw_sr[tt] = cw_sr[tt - 1] * Re_sr[tt - 1]
    cw_rr[tt] = cw_rr[tt - 1] * Re_rr[tt - 1]


xcv = numpy.arange(0, T+1)

# plot Cumulative Wealth
plt.plot(xcv, cw_1n, color='c', linestyle='-.', marker='+', markersize=4, label="Market")
plt.plot(xcv, cw_si, color='g', linestyle='-.', marker='^', markersize=4, label="Best-Stock")
plt.plot(xcv, cw_vv, color='b', linestyle='--', marker='p', markersize=4, label="Min_V")
plt.plot(xcv, cw_sr, color='k', linestyle=':', marker='x', markersize=4, label="Max_SR")
plt.plot(xcv, cw_o, color='y', linestyle='--', marker='d', markersize=4, label="E2E-without")
plt.plot(xcv, cw_b, color='r', linestyle='-', marker='*', markersize=4, label="E2E-with")
plt.xlabel('Day')
plt.ylabel('Cumulative Wealth')
plt.legend(loc='best')
plt.subplots_adjust(top=0.9, bottom=0.1, right=0.9, left=0.1)
plt.show()

# Other Results
ret = torch.stack([R_1n, Re_si, Re_vv, Re_sr, Re_o, torch.from_numpy(Re_b)], 1)-1
cw = torch.stack([torch.from_numpy(cw_1n), torch.from_numpy(cw_si), torch.from_numpy(cw_vv), torch.from_numpy(cw_sr), torch.from_numpy(cw_o), torch.from_numpy(cw_b)], 1)
ret = ret.detach()
ret_m = ret.mean(0)
ret_v = ret.std(0)
# ret_sr = (ret_m*242 - 0.0015)/(ret_v*numpy.sqrt(242))
ret_sr = ret_m/ret_v

ca = torch.zeros(T, 6)
for tt in range(1, T+1):
    cmax = (cw[0:tt+1]).max(0).values
    ca[tt-1] = (cmax - cw[tt]) / cmax
    # print(ca[tt-1])

cav = ca.max(0).values
ret_cav = ret_m/cav
b_s = torch.tensor([1, 2, 3, 4, 5])
a = ret[:, 0]
b = ret[:, b_s]
a_m = a.mean(0)
b_m = b.mean(0)
MER = b_m - a_m

be = torch.zeros(5)
al = torch.zeros(5)
ee = torch.zeros(T, 5)

for ji in range(5):
    be[ji] = ((a-a_m)*(b[:, ji]-b_m[ji])).sum()/((a-a_m).pow(2).sum())
    al[ji] = b_m[ji] - be[ji] * a_m
    ee[:, ji] = b[:, ji] - (al[ji] + be[ji]*a)

sd_ee = ee.std(0)
sd_al = sd_ee * ((a.pow(2).sum())/((a-a_m).pow(2).sum()*T)).sqrt()

tv = al/sd_al
nor = tdist.Normal(torch.tensor([0]), torch.tensor([1]))

pv = 2*(1 - nor.cdf(tv.abs()))

ret_mm = (b - a.unsqueeze(1))
ret_ir = ret_mm.mean(0) / ret_mm.std(0)

# Print Results
# Market(opt), Best-Stock, Min_V, Min_Sr, E2E_without, E2E_with
print('Best-Stock', 'Min_V', 'Min_Sr', 'E2E_without', 'E2E_with')
print('MER:', MER)
print('beta:', be)
print('alpha:', al)
print('tv:', tv)
print('pv:', pv)
print('Market', 'Best-Stock', 'Min_V', 'Min_Sr', 'E2E_without', 'E2E_with')
print('sr:', ret_sr)
print('ir:', ret_ir)
print('cr:', ret_cav)
