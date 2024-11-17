# catradar
<img src=https://github.com/user-attachments/assets/8430ef08-1c38-43b6-b5d8-1d1980ac0bd2 alt="logo" width="100" align="right">

Catradar — инструмент для запуска симуляции взаимодействия объектов (далее котов) на ограниченной карте и логгирования их взаимодействия.

## Правила взаимодействия
Взаимодействие котов между собой определяется по следующим правилам:
1. Если два кота находятся на расстоянии не превышающем `R0`, то они пытаются начать драку с вероятностью 1.

2. Если два кота находятся на расстоянии `R1`, там, что `R1 > R0`, они начинают шипеть с вероятностью обратно пропорциональной квадрату расстояния между ними.

3. Если вокруг кота нет соперников он перемещается согласно текущему правилу.

Перемещение котов картой размера `X` на `Y`

## Возможности

На текущий момент приложение:
1. Обрабатывает и отрисовывает более чем 1 миллион котов на карте за 33 мс
2. Логгирует и подсвечивает взаимодействие животных
3. Предоставляет три паттерна движения
    - Свободное
    - Карусельное
    - С обработкой столкновений
4. Позволяет влиять на движение посредством курсора
5. Позволяет регулировать следущие параметры без перезапуска симуляции:
    - Выбор коэффициента отрисовки
    - Установку пресета для начальной позиции
    - Выбор паттерна движения
    - Изменение скорости движения
    - Выбор функции расстояния взаимодействия
6. Возможность осмотреть карту

### Демонстрация

#### Выбери своё движение
##### Свободное

![free](https://github.com/user-attachments/assets/7b9149fc-882d-4d1a-9edc-6534c3478c0c)

##### Карусельное


![carousel](https://github.com/user-attachments/assets/5808ea39-8bed-43f3-88db-2d352a3ffde9)

##### С обработкой столкновений

![collide](https://github.com/user-attachments/assets/d14b1c38-56dc-433d-9be3-554005dc9133)

#### Толкай и изменяй скорость

![push-and-speed](https://github.com/user-attachments/assets/11d9b564-c36e-4112-9504-ff273696dfe3)

#### Меняй правила симуляции

https://github.com/user-attachments/assets/0a0f4b01-9644-42ff-a1ea-e1408c327f3b


## Для разработчиков
### Начальная настройка
```bash
rye sync && rye run pre-commit install
```

### Запуск
```bash
rye run python -m src.catradar
```

### Запуск тестов
```bash
rye test
```

## Использованные инструменты

Для реализации симуляции и GUI был выбран [taichi](https://github.com/taichi-dev/taichi).
```quote
Taichi Lang is an open-source, imperative, parallel programming language for high-performance numerical computation. It is embedded in Python and uses just-in-time (JIT) compiler frameworks, for example LLVM, to offload the compute-intensive Python code to the native GPU or CPU instructions.
```

## Лицензия
Код распространяется под лицензией MIT. Подробнее в файле [LICENCE](./LICENCE).
