import pandapower.networks as nw
from pandapower.plotting import simple_plot, simple_plotly, pf_res_plotly

net = nw.mv_oberrhein()
net = nw.ieee_european_lv_asymmetric()
simple_plotly(net)

