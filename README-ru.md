# $\lambda$-СКА как архитектурно интерпретируемый ИИ.

![gif1](MNIST_example\logs\MnistLR\lightning_logs\version_0\sens_pic\MnistLR.gif)
![gif2](MNIST_example\logs\Mnist_tlmdSplineKAN\lightning_logs\version_0\sens_pic\tlmdSplineKAN.gif)
*Изменение карт чувствительности линейного классификатора и обучаемой $\lambda$-СКА в процессе обучения*

# Общее описание
В этом проекте представлен программный код, реализующий $\lambda$ - сеть Колмогорова-Арнольда ($\lambda$-СКА).

$\lambda$-СКА - это модификация СКА, основанная на предложенной Кахане версии теоремы Колмогорова-Арнольда:
$$f(x_{1},\cdot\cdot\cdot, x_{n})=\sum_{q=1}^{2n+1}\Phi_{q}(\sum_{p=1}^{n}\lambda_{p}\cdot\varphi_{q}(x_{p}))$$
Данная модификация СКА позволяет рассматривать коэффициенты $\lambda_{p}$ как меру чувствительности модели к входным признаккам $x_{p}$.

$\lambda$-СКА была протестирована на 4 синтетических, 2 реальных наборах данных и MNIST с использованием SHAP и LIME в качестве референсных методов интерпретации.

![pic1](pictures\pic1.png)
*SHAP, LIME and $\lambda$ sensitivity analyses of $\lambda$-KAN on Curated solubility and Boston housing datasets*
*Анализ чувствительности $\lambda$-СКА методами SHAP, LIME и $\lambda$-коэффициентов на бенчмарках Curated solubility (сверху) и Boston housing (снизу)*


---
С целью улучшения аппроксимационных возможностей при сохранении интерпретационных возможностей $\lambda$-СКА, была введена обучаемая $\lambda$-СКА:
$$f(x_{1},\cdot\cdot\cdot, x_{n})=\sum_{q=1}^{2n+1}\Phi_{q}(\sum_{p=1}^{n}\lambda_{p}(1+\alpha g(\bold{x}))\cdot\varphi_{q}(x_{p}))$$


Обучаемая $\lambda$-СКА показала наилучший баланс интерпретируемость-точность на задаче MNIST.

|  | Лин. класс. | $\lambda$-СКА | Обучаемая $\lambda$-СКА | MLP |
|--|-------------|---------------|-------------------------|-----|
| Точность на тест. н. | 92.5% | 89.2% | 95.9% | 97.9% |


Код был частично основан на фреймворке pykan (https://github.com/KindXiaoming/pykan?ysclid=mbpkse6kxx388413783)

---
# Файлы кода

1) Вспомогательный код:
    * source.py - 
        1. Визуализация функций и анализ результатов СКА; 
        2. Реализация $\lambda$-СКА;
        3. Анализ интерпретируемости $\lambda$-СКА.


2) _Benchmarks_/: Обучение и анализ интерпретируемости $\lambda$-СКА с использованием тестовых функций и бенчмарков реального мира.
    * lmd_kan_f1.ipynb
    * lmd_kan_f1_extend_input.ipynb - _extended_input_  означает добавление двух случайных входных признаков.

        ...
    * lmd_kan_f4.ipynb
    * lmd_kan_f4_extend_input.ipynb
    * lmd_kan_boston_housing.ipynb - используется бенчмарк Boston housing.
    * lmd_kan_curated_solubility - используется бенчмарк Curated solubility.


3) _mnist_example_/: Применение многослойной нейронной сети, линейного классификатора, $\lambda$-СКА и обучаемой $\lambda$-СКА к задаче MNIST. Исследование интерпретируемости трех последних моделей.

    Gif-изображения эволюции карт чувствительности в процессе обучения можно найти в _MNIST_example/logs/.../sens_pic_
---