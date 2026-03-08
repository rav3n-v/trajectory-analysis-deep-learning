import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white", palette="pastel")

x = np.linspace(0.2,10,100)
plt.plot(x, x**3)

plt.xticks(visible=False)
plt.yticks(visible=False)




plt.show()