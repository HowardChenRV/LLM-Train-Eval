import argparse


def add_aurora_args(parser):
    group = parser.add_argument_group(title='Data-Client')

    group.add_argument('--aurora-test-type', default=None,
                       choices=["training/pretrain"],
                       help='Test type for data client',)

    group.add_argument('--aurora-save-dir', default=None,
                       help='local data store path for data client',)

    group.add_argument('--aurora-tester', default='tester',
                       help='Tester for data client',)
    
    group.add_argument('--aurora-hardware-name', default=None,
                       help='Hardware name for data client',)
    
    group.add_argument('--aurora-platform-provider', default='cloud',
                        help='Platform provider for data client',)
    
    group.add_argument('--aurora-model-serial', default='gpt',
                        help='Model serial for data client',)
    
    group.add_argument('--aurora-model-size', default=0, type=int,
                        help='Model size for data client',)
    
    group.add_argument('--aurora-framework-name', default='Megatron-LM',
                        help='Framework name for data client',)
    
    group.add_argument('--aurora-framework-version', default='0.8.0',
                        help='Framework version for data client',)
    
    group.add_argument('--aurora-test-source', default='test',
                        help='Test source for data client',)
    
    return parser
