from scipy.stats import beta
from scipy.special import logit
from scipy.special import binom
import matplotlib.pyplot as plt
import numpy as np

# Define plotting parameters
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20
lw = 2
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE, linewidth=lw)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('xtick.major', width=lw)
plt.rc('xtick.minor', width=0.5*lw)
plt.rc('ytick.major', width=lw)
plt.rc('ytick.minor', width=0.5*lw)
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('lines', linewidth=lw*1.25)
plt.rc('figure', autolayout=True)
plt.rc('legend', fontsize=MEDIUM_SIZE)


# Initialize beta parameters
a = 10**-1
b = 10**-1
order = 30

# Plot beta distribution
x = []
val = []
for i in range(100+1):
    x.append(i)
    mom = lambda y: y**i*(1-y)**(order+2-i)
    val.append(beta.pdf(i/100, a, b))
plt.plot(np.array(x)/(100), val, label='PDF ')

plt.xlabel('Frequency of A in Population')
plt.ylabel('PDF')
plt.savefig('beta_pdf.pdf')
plt.close()


#####################################################################
# This section generates plot 1(a)




# Plot n, n+1, n+2 on top of one another
# Plot moments of order n
x = []
val = []
for i in range(order+1):
    x.append(i)
    mom = lambda y: y**i*(1-y)**(order-i)
    val.append(binom(order,i)*beta.expect(mom, args=(a, b)))
plt.plot(np.array(x)/order, val, label='Order ' + str(order), marker='x')

# Plot moments of order n+1
x = []
val = []
for i in range(order+2):
    x.append(i)
    mom = lambda y: y**i*(1-y)**(order+1-i)
    val.append(binom(order+1,i)*beta.expect(mom, args=(a, b)))
plt.plot(np.array(x)/(order+1), val, label='Order ' + str(order+1), marker='x')

# Plot moments of order n+2
x = []
val = []
for i in range(order+3):
    x.append(i)
    mom = lambda y: y**i*(1-y)**(order+2-i)
    val.append(binom(order+2,i)*beta.expect(mom, args=(a, b)))
plt.plot(np.array(x)/(order+2), val, label='Order ' + str(order+2), marker='x')

plt.legend()
plt.xlim(0, 1)
plt.xlabel('Frequency of A in Sample')
plt.ylabel('Moment')
plt.savefig('beta_ut.pdf')
plt.close()



#####################################################################
# This section generates plot 1(b)



# Plot n, n+1, n+2 on top of one another zoomed in
# Plot moments of order n
x = []
val = []
for i in range(3, order-2):
    x.append(i)
    mom = lambda y: y**i*(1-y)**(order-i)
    val.append(binom(order,i)*beta.expect(mom, args=(a, b)))
plt.plot(np.array(x)/order, val, label='Order ' + str(order), marker='x')

# Plot moments of order n+1
x = []
val = []
for i in range(3, order-1):
    x.append(i)
    mom = lambda y: y**i*(1-y)**(order+1-i)
    val.append(binom(order+1,i)*beta.expect(mom, args=(a, b)))
plt.plot(np.array(x)/(order+1), val, label='Order ' + str(order+1), marker='x')

# Plot moments of order n+2
x = []
val = []
for i in range(3, order):
    x.append(i)
    mom = lambda y: y**i*(1-y)**(order+2-i)
    val.append(binom(order+2,i)*beta.expect(mom, args=(a, b)))
plt.plot(np.array(x)/(order+2), val, label='Order ' + str(order+2), marker='x')
plt.xlim(.2, .8)
plt.ylim(.005, .0095)

plt.legend()
plt.xlabel('Frequency of A in Sample')
plt.ylabel('Moment')
plt.savefig('beta_utzoom.pdf')
plt.close()




#####################################################################
# This section generates plot 1(c)


# Plot re-scaled momdents on same plot
# Plot re-scaled moemnts of order n
x = []
val = []
for i in range(3, order-2):
    x.append(i)
    mom = lambda y: y**i*(1-y)**(order-i)
    val.append(binom(order,i)*beta.expect(mom, args=(a, b)))
val = np.array(val)
plt.plot(np.array(x)/order, (order + 1) / (order + 3) * val, label='Order ' + str(order), marker='x')

# Plot re-scaled moemnts of order n+1
x = []
val = []
for i in range(3, order-1):
    x.append(i)
    mom = lambda y: y**i*(1-y)**(order+1-i)
    val.append(binom(order+1,i)*beta.expect(mom, args=(a, b)))
val = np.array(val)
plt.plot(np.array(x)/(order+1), (order + 2) / (order + 3) * val, label='Order ' + str(order+1), marker='x')

# Plot moemnts of order n+2
x = []
val = []
for i in range(3, order):
    x.append(i)
    mom = lambda y: y**i*(1-y)**(order+2-i)
    val.append(binom(order+2,i)*beta.expect(mom, args=(a, b)))
val = np.array(val)
plt.plot(np.array(x)/(order+2), val, label='Order ' + str(order+2), marker='x')

plt.xlim(.2, .8)
plt.ylim(.005, .0095)
plt.legend()
plt.xlabel('Frequency of A in Sample')
plt.ylabel('Re-scaled Moment')
plt.savefig('beta_t1.pdf')
plt.close()





#####################################################################
# This section generates plot 1(d)




# Plot logit re-scaled momdents on same plot
# Plot logit re-scaled moments of order n
x = []
val = []
for i in range(order+1):
    x.append(i)
    mom = lambda y: y**i*(1-y)**(order-i)
    val.append(binom(order,i)*beta.expect(mom, args=(a, b)))
val = np.array(val)
plt.plot(np.array(x)/order, logit((order + 1) / (order + 3) * val), label='Order ' + str(order), marker='x')

# Plot logit re-scaled moments of order n+1
x = []
val = []
for i in range(order+2):
    x.append(i)
    mom = lambda y: y**i*(1-y)**(order+1-i)
    val.append(binom(order+1,i)*beta.expect(mom, args=(a, b)))
val = np.array(val)
plt.plot(np.array(x)/(order+1), logit((order + 2) / (order + 3) * val), label='Order ' + str(order+1), marker='x')

# Plot moemnts of order n+2
x = []
val = []
for i in range(order+3):
    x.append(i)
    mom = lambda y: y**i*(1-y)**(order+2-i)
    val.append(binom(order+2,i)*beta.expect(mom, args=(a, b)))
val = np.array(val)
plt.plot(np.array(x)/(order+2), logit(val), label='Order ' + str(order+2), marker='x')

plt.xlim(.2, .8)
plt.ylim(-5.25, -4.7)
plt.legend()
plt.xlabel('Frequency of A in Sample')
plt.ylabel('Re-scaled Logit Moment')
plt.savefig('beta_t2a.pdf')
plt.close()