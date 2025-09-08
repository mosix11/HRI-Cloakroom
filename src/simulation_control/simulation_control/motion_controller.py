#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer

from interfaces.action import StoreItems, ReturnItems

class MotionController(Node):
    def __init__(self):
        super().__init__('motion_controller')
        self.get_logger().info('Motion Controller Node Initialized')
        
        
        self._item_storage_action_server = ActionServer(
            self,
            StoreItems,
            'motion_cotroller/store_items',
            self.store_items
        )
        
        self._item_retrieval_action_server = ActionServer(
            self,
            ReturnItems,
            'motion_cotroller/return_items',
            self.return_items
        )
        
        
   
    def store_items(self, goal_handle):
        self.get_logger().info('Executing Store Action ...')
        
            
        goal_handle.succeed()
        result = StoreItems.Result()
        result.success = True
        result.message = ''
        return result
        
        
    def return_items(self, goal_handle):
        self.get_logger().info('Executing Return Action ...')
        
            
        goal_handle.succeed()
        result = ReturnItems.Result()
        result.success = True
        result.message = ''
        return result




def main(args=None):
    rclpy.init(args=args)
    node = MotionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt, shutting down Motion Controller node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
