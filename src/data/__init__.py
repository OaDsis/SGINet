from importlib import import_module
from dataloader import MSDataLoader

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())     # 等同于：import data.rainheavy
            trainset = getattr(module_train, args.data_train)(args)     # 等同于：trainset = RainHeavy(args)
            self.loader_train = MSDataLoader(           # 初始化训练数据
                args,
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu
            )

        if args.data_test in ['Set5', 'Set14', 'B100', 'Urban100']:
            module_test = import_module('data.benchmark')
            testset = getattr(module_test, 'Benchmark')(args, train=False)
        else:
            module_test = import_module('data.' +  args.data_test.lower())      # 等同于：import data.rainheavytest
            testset = getattr(module_test, args.data_test)(args, train=False)   # 等同于：testset = RainHeavyTest(args)

        self.loader_test = MSDataLoader(                # 初始化测试数据
            args,
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu
        )

