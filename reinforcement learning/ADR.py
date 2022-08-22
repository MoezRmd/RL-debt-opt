import random

import numpy as np
import pandas as pd


data = pd.read_csv("C:/Users/Moez/Documents/en_cours.csv",encoding = "ISO-8859-1",sep=',')
features = ["motivation",
            "pricipal",
            "rest_principal",
            "interest",
            "rest_interest",
            "total_vers",
            "nbre versement",
            "income",
            "age",
            "house",
            "married",
            "children",
            "term",
            "installment"
            ]
Amiable=0
Precontentieux=1
Judiciel=2
Non_permession = 3
step_pen = -1
prec_pen = -100
jud_pen = -10000
class ADREnv():
    def __init__(self):
        self.observation = dict.fromkeys(features)
        self.horizon = 70
        self.reward = 0
        self.done = False
        self.a = 0
        self.pc = 0
        self.j = 0
        self.illegal_raward = -100
        self.reset()
    def reset(self):
        # self.observation = dict.fromkeys(features)
        doss = data.iloc[random.randint(0, data.shape[0])]
        #self.observation['group']=doss['Groupe']
        self.observation['pricipal'] = doss['Principal']
        self.observation['rest_principal'] = doss['solde Principal']
        self.observation['interest'] = doss['Solde Interets']
        self.observation['rest_interest'] = doss['Interet restant']
        self.observation['total_vers'] = doss['total_vers']
        self.observation['nbre versement'] = doss['nbre versement']
        self.observation['motivation'] = random.randint(0,1)
        self.observation['age'] = random.randint(20, 65)
        self.observation['house'] = random.randint(0, 1)
        self.observation['married'] = random.randint(0, 1)
        self.observation['children'] = random.randint(0, 5)
        self.observation['income'] = random.uniform(1000, 2500)

        self.observation['term'] = random.choice([36,48,60,72,120])
        self.observation['installment'] = (doss['Solde Interets']+doss['Principal'])/self.observation['term']
        self.reward = 0
        self.done = (self.observation['rest_principal'] == 0.0)
        self.a=10
        self.pc=2
        self.j=1
        return self.observation

    def step(self,action):

        # a=10
        # pc=5
        # j=2
        payment_p = 0
        payment_i = 0
        if self.done:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        self.reward = 0
        if (self.a != 0 & action != 0) | (self.a == 0 & self.pc != 0 & action != 1 ) | (self.a == 0 & self.pc == 0 & action != 2 ):
            self.reward += self.illegal_raward
            print('wrong action!!')
        else:
            self.reward += step_pen
            if action == 0:  #amiable
                self.observation['motivation'] = random.randint(0,1)
                intuation = random.randint(0,1)
                if intuation == 1 : #  willing to pay
                    if self.observation['rest_principal'] < self.observation['installment']:
                        payment_p = self.observation['rest_principal']
                        payment_i = random.uniform(0,self.observation['rest_interest'])


                    else :
                        payment_p = random.uniform(self.observation['installment'], self.observation['rest_principal'])
                        payment_i = random.uniform(0, self.observation['rest_interest'])

                    self.reward += payment_p+payment_i
                    self.observation['rest_principal']-=payment_p
                    self.observation['rest_interest'] -= payment_i
                    self.observation['nbre versement'] += 1
                    self.observation['total_vers'] += payment_p + payment_i

                self.a += -1

            if action == 2 :
                self.reward += jud_pen
                self.done = True
                self.j += -1
            if self.observation['rest_principal'] == self.observation['rest_interest'] == 0.0:
                self.done = True


            if action == 1 : #phase precontentieux

                self.observation['motivation'] = random.randint(0, 1)
                intuation = random.randint(0, 1)
                if intuation == 1:  # willing to pay
                    if self.observation['rest_principal'] < self.observation['installment']:
                        payment_p = self.observation['rest_principal']
                        payment_i = random.uniform(0, self.observation['rest_interest'])


                    else:
                        payment_p = random.uniform(self.observation['installment'],
                                                   self.observation['rest_principal'])
                        payment_i = random.uniform(0, self.observation['rest_interest'])

                    self.reward += prec_pen
                    self.observation['rest_principal'] -= payment_p
                    self.observation['rest_interest'] -= payment_i
                    self.observation['nbre versement'] += 1
                    self.observation['total_vers'] += payment_p + payment_i

                self.pc += -1
        return np.array(self.observation), self.reward, self.done, None



