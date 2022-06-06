import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# root là nơi tải dữ liệu xuống
# train = true, trả về tập dữ liệu về datasets
# biến đổi thực hiện một số xử lý trước hữu ích
# Tải dữ liệu từ mnist
train_dataset = torchvision.datasets.MNIST(root='.', train=True, transform=transforms.ToTensor(), download=True)
train_dataset.data
train_dataset.data.max() # tensor(255, dtype=torch.uint8)
train_dataset.data.shape # torch.Size([60000, 28, 28])
train_dataset.targets # between 0 and 9

#--------------------------------
learning_rate = 0.03 # Tỉ lệ học |
#--------------------------------

# tập dữ liệu kiểm tra
test_dataset = torchvision.datasets.MNIST(root='.', train=False, transform=transforms.ToTensor(), download=True)

#---------------------------------------------
#              xây dựng mô hình              |
#---------------------------------------------
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(30), nn.Linear(128, 10))



# không cần đến softmax cuối cùng!
device = torch.device("cude:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Thất bại và tối ưu hóa
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# Tải dữ liệu
# Hữu ích vì nó tự động tạo các ô trong vòng huấn luyện
# và quan tâm đến việc xáo trộn

batch_size = 128
# xáo trộn dữ liệu đào tạo, nhưng không trộn dữ liệu kiểm tra
# Em không muốn có sự tương quan giữa các dữ liệu
# không cần xáo trộn dữ liệu thử nghiệm
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Kiểm tra những gì trình tải dữ liệu làm
# ánh xạ các giá trị thành (0, 1)
# tạo dữ liệu về hình dạng (kích thước lô, màu sắc, chiều cao, chiều rộng)
tmp_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

for x,y in tmp_loader:
  print(x)
  print(x.shape)
  print(y.shape)
  break
train_dataset.transform(train_dataset.data.numpy()).max()
# Số lần huấn luyện mô hình
n_epochs = 10

# Nội dung cần lưu trữ
train_losses = np.zeros(n_epochs)
test_losses = np.zeros(n_epochs)


for it in range(n_epochs):
  train_loss = []
  for inputs,targets in train_loader:
 	# Di chuyển dữ liệu sang GPU nếu thiết bị là GPU
    # if device == 'cpu':
    inputs, targets = inputs.to(device), targets.to(device)

    # Định hình lại đầu vào
    # -1 nghĩa là chỉ định bất kỳ giá trị nào phù hợp
    inputs = inputs.view(-1, 784)
     
    # Zero the parameter gradients     
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward and Optimize
    loss.backward()
    optimizer.step()

    train_loss.append(loss.item())

      

  # Nhận tỉ lệ thất bại huấn luyện và kiểm tra thất bại
  train_loss = np.mean(train_loss)  
  test_loss = []
  for inputs, targets in test_loader:
    inputs, targets = inputs.to(device), targets.to(device)
    inputs = inputs.view(-1, 784)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    test_loss.append(loss.item())
  test_loss = np.mean(test_loss)

  # Save losses
  train_losses[it] = train_loss
  test_losses[it] = test_loss
  
  print(f'Epoch {it+1}/{n_epochs}, Tỉ lệ thất bại của dử liệu huấn luyện: {train_loss:.4f}, Tỉ lệ thất bại của tập dử liệu kiểm tra: {test_loss:.4f}')

  # Vẽ biểu đồ tỉ lệ thất bại
  plt.plot(train_losses, label='Tỉ lệ thất bại của tập huấn luyện ')
  plt.plot(test_losses, label='Tỉ lệ thất bại của tập kiểm tra')
  plt.legend()
  plt.show()

# Tính độ chính xác

# độ chính xác của huấn luyện
n_correct = 0
n_total = 0
for inputs, targets in train_loader:
  # di chuyển dữ liệu sang GPU
  inputs, targets = inputs.to(device), targets.to(device)

  # định hình lại đầu vào
  inputs = inputs.view(-1, 784)

  # chuyển tiếp qua
  outputs = model(inputs)

  # nhận dự đoán
  # torch.max trả về cả max và argmax
  _, predictions = torch.max(outputs, 1)

  # cập nhật số lượng
  n_correct += (predictions == targets).sum().item()
  n_total += targets.shape[0]

train_acc = n_correct / n_total

# test accuracy
n_correct = 0
n_total = 0
for inputs, targets in test_loader:
  # duy chuyển dữ liệu vào GPU
  inputs, targets = inputs.to(device), targets.to(device)

  # định hình lại đầu vào
  inputs = inputs.view(-1, 784)

 # chuyển tiếp qua
  outputs = model(inputs)

 # nhận dự đoán
 # torch.max trả về cả max và argmax
  _, predictions = torch.max(outputs, 1)

  # cập nhật số lượng
  n_correct += (predictions == targets).sum().item()
  n_total += targets.shape[0]

test_acc = n_correct / n_total


print(f'Độ chính xác của tập dữ liệu huấn luyện: {train_acc:.4f}, Độ chính xác của tập dử liệu kiểm tra: {test_acc:.4f}')

#Ma trận hỗn loạn
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
  """
  The function prints and plots the confusion matrix.
  Normalization can be applied by setting 'normalize=True'
  """

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Chuẩn hoá ma trận hỗn loạn")

  else:
    print('Ma trận hỗn loạn, không chuẩn hóa')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'

  thresh = cm.max() / 2
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('Nhãn thật')
  plt.xlabel('Nhãn dự đoán')
  plt.show()

# get all predictions in an array and plot confusion matrix
x_test = test_dataset.data.numpy()
y_test = test_dataset.targets.numpy()
p_test = np.array([])
for inputs, targets in test_loader:
  # move data to GPU
  inputs, targets = inputs.to(device), targets.to(device)

  # reshape the input
  inputs = inputs.view(-1, 784)

  # forward pass
  outputs = model(inputs)

  # get the prediction
  #torch.max returns both max and argmax
  _, predictions = torch.max(outputs, 1)

  # update p_test
  p_test = np.concatenate((p_test, predictions.cpu().numpy()))

cm = confusion_matrix(y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))


# Thử hiển thi một số ví dụ đúng
print("-------------------------------------------------------------")
countss = 1
while (countss < 16):
  print ('Ảnh dự đoán đúng:', countss)
  countss = countss + 1
  misclassified_idx = np.where(p_test == y_test)[0]
  i = np.random.choice(misclassified_idx)
  plt.imshow(x_test[i], cmap='gray')
  plt.title("Giá trị thật: %s Dự đoán: %s" % (y_test[i], int(p_test[i])))
  plt.show()

# Thử hiển thị một số ví dụ phân loại sai

print("-------------------------------------------------------------")
counts = 1
while (counts <= 5):
  print ('Ảnh dự đoán sai:', counts)
  counts = counts + 1
  misclassified_idx = np.where(p_test != y_test)[0]
  i = np.random.choice(misclassified_idx)
  plt.imshow(x_test[i], cmap='gray')
  plt.title("Giá trị thật: %s Dự đoán: %s" % (y_test[i], int(p_test[i])))
  plt.show()
  #////