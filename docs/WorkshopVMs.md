# Workshop Virtual Machines instruction
All the work done for the PSIML workshops will be done in Azure VMs.

## Connecting to a VM
To connect to a VM, follow these steps (assuming you have a Windows machine. Steps are similar on different OS):
1. Find your name in the table below, and find your machine details.
2. Open Remote Desktop Connection by clicking the Start button Start button icon. In the search box, type Remote Desktop Connection, and then, in the list of results, click Remote Desktop Connection.
3. In the Computer box, type the **DNS name** of the computer that you want to connect to, and then click Connect.
4. type in username and password per table below and click OK.

### Linux notes:
Linux does not come with RDP viewer by default. To be able to log in one needs to be installed. There are several options available (Remmina, rdesktop, freerdp etc). Please install the RDP viewer using the package manager for your Linux distribution. For  Debian based distributions and Remmina steps are outlined below:
```
sudo apt-get update
sudo apt-get install remmina reminna-plugin-rdp
```
Once installed, start Remmina, click + button, fill in the required credentials (username, domain, password) and click Connect.

## Starting the workshop
To start the workshop:
1. find the cloned git repository in c:\PSIML
2. Click the Start button icon. In the search box, type Anaconda Prompt, and then, in the list of results, click Anaconda Prompt.
3. Activate the AzureML environment by typing: `conda activate AzureML` in Anaconda Prompt
3. follow the workshop lead instructions


## List of participants and VM details
|participant            | machine name  |   DNS name                                  | username                | password     |
|-----------------------|---------------|---------------------------------------------|-------------------------|--------------|
|Aleksa Milisavljević   | zumra-WS-VM1  | zumra-ws-vm1.westeurope.cloudapp.azure.com  | zumra-ws-vm1\psimluser  | Petnica2019  |
|Aleksandra Stevanović  | zumra-WS-VM2  | zumra-ws-vm2.westeurope.cloudapp.azure.com  | zumra-ws-vm2\psimluser  | Petnica2019  |
|Anastasija Ilić        | zumra-WS-VM3  | zumra-ws-vm3.westeurope.cloudapp.azure.com  | zumra-ws-vm3\psimluser  | Petnica2019  |
|Haris Gegić            | zumra-WS-VM4  | zumra-ws-vm4.westeurope.cloudapp.azure.com  | zumra-ws-vm4\psimluser  | Petnica2019  |
|Irena Đorđević         | zumra2-WS-VM5  | zumra2-ws-vm5.westeurope.cloudapp.azure.com  | zumra2-ws-vm5\psimluser  | Petnica2019  |
|Ivan Pop-Jovanov       | zumra2-WS-VM6  | zumra2-ws-vm6.westeurope.cloudapp.azure.com  | zumra2-ws-vm6\psimluser  | Petnica2019  |
|Jelena Ristić          | zumra2-WS-VM7  | zumra2-ws-vm7.westeurope.cloudapp.azure.com  | zumra2-ws-vm7\psimluser  | Petnica2019  |
|Kosta Grujčić          | zumra2-WS-VM8  | zumra2-ws-vm8.westeurope.cloudapp.azure.com  | zumra2-ws-vm8\psimluser  | Petnica2019  |
|Kosta Jovanović        | zumra2-WS-VM9  | zumra2-ws-vm9.westeurope.cloudapp.azure.com  | zumra2-ws-vm9\psimluser  | Petnica2019  |
|Luka Jovičić           | PSIML-WS-VM10 | psiml-ws-vm10.westeurope.cloudapp.azure.com | psiml-ws-vm10\psimluser | Petnica2019  |
|Marina Vasiljević      | PSIML-WS-VM11 | psiml-ws-vm11.westeurope.cloudapp.azure.com | psiml-ws-vm11\psimluser | Petnica2019  |
|Marko Ložajić          | PSIML-WS-VM12 | psiml-ws-vm12.westeurope.cloudapp.azure.com | psiml-ws-vm12\psimluser | Petnica2019  |
|Mihailo Grbić          | PSIML-WS-VM13 | psiml-ws-vm13.westeurope.cloudapp.azure.com | psiml-ws-vm13\psimluser | Petnica2019  |
|Miloš Stanković        | PSIML-WS-VM14 | psiml-ws-vm14.westeurope.cloudapp.azure.com | psiml-ws-vm14\psimluser | Petnica2019  |
|Nikola Bebić           | PSIML-WS-VM15 | psiml-ws-vm15.westeurope.cloudapp.azure.com | psiml-ws-vm15\psimluser | Petnica2019  |
|Nikola Spasić          | PSIML-WS-VM16 | psiml-ws-vm16.westeurope.cloudapp.azure.com | psiml-ws-vm16\psimluser | Petnica2019  |
|Ognjen Milinković      | PSIML-WS-VM17 | psiml-ws-vm17.westeurope.cloudapp.azure.com | psiml-ws-vm17\psimluser | Petnica2019  |
|Ognjen Nenezić         | PSIML-WS-VM18 | psiml-ws-vm18.westeurope.cloudapp.azure.com | psiml-ws-vm18\psimluser | Petnica2019  |
|Pavle Divović          | PSIML-WS-VM19 | psiml-ws-vm19.westeurope.cloudapp.azure.com | psiml-ws-vm19\psimluser | Petnica2019  |
|Radenko Pejić          | PSIML-WS-VM20 | psiml-ws-vm20.westeurope.cloudapp.azure.com | psiml-ws-vm20\psimluser | Petnica2019  |
|Radoica Draskić        | PSIML-WS-VM21 | psiml-ws-vm21.westeurope.cloudapp.azure.com | psiml-ws-vm21\psimluser | Petnica2019  |
|Slobodan Jenko         | PSIML-WS-VM22 | psiml-ws-vm22.westeurope.cloudapp.azure.com | psiml-ws-vm22\psimluser | Petnica2019  |
|Stefan Stepanović      | PSIML-WS-VM23 | psiml-ws-vm23.westeurope.cloudapp.azure.com | psiml-ws-vm23\psimluser | Petnica2019  |
|Vuk Radović            | PSIML-WS-VM24 | psiml-ws-vm24.westeurope.cloudapp.azure.com | psiml-ws-vm24\psimluser | Petnica2019  |
|Đordje Vasiljević      | PSIML-WS-VM25 | psiml-ws-vm25.westeurope.cloudapp.azure.com | psiml-ws-vm25\psimluser | Petnica2019  |
