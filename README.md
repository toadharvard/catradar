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
1. Обрабатывает и отрисовывает более чем 1 миллион котов на карте за 50 мс
2. Логгирует и подсвечивает их взаимодействие
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
6. Возможность осмотреть карту c помощью `WASD`, приблизить и удалиться на `Q` и `E`
7. Ограничение "количество котов меняющих за один период статус не превышает 250" заменено более слабое "количество котов в одной клетке виртуальной сетки карты с ребром `R1` не более `LIMIT`"
8. [Поддержка Metal, OpenGL, Vulkan в качестве бекенда, а так же Mac OS на M1, Intel и Windows и Linux на x86_64](https://docs.taichi-lang.org/docs/hello_world#supported-systems-and-backends)
9. Поддержка вида от третьего лица для наблюдения за конкретным котом
10. Поддержка барьеров-препятствий для котов

## Виды движения
### Свободное

![free](https://github.com/user-attachments/assets/7b9149fc-882d-4d1a-9edc-6534c3478c0c)

Наиболее простая форма движения
### Карусельное

![carousel](https://github.com/user-attachments/assets/5808ea39-8bed-43f3-88db-2d352a3ffde9)

Коты кружат в вальсе, образуя незамысловатые фигуры
### С обработкой столкновений

![collide](https://github.com/user-attachments/assets/d14b1c38-56dc-433d-9be3-554005dc9133)

Если коты наталкиваются друг на друга, то сразу же разбегаются в разные стороны, избегая драки

![cursor-push](https://github.com/user-attachments/assets/32563493-9050-4588-aebd-0ced209ff7f7)

Клик по области с включенной опцией `Cursor Push` позволяет напугать котов и отолкнуть от себя

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

Для реализации симуляции и GUI был выбран [taichi](https://github.com/taichi-dev/taichi). Вот так авторы описывают свой инструмент:

Taichi Lang is an open-source, imperative, parallel programming language for high-performance numerical computation. It is embedded in Python and uses just-in-time (JIT) compiler frameworks, for example LLVM, to offload the compute-intensive Python code to the native GPU or CPU instructions.

## Лицензия
Код распространяется под лицензией MIT. Подробнее в файле [LICENCE](./LICENCE).
