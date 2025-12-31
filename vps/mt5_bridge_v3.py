#!/usr/bin/env python3
"""
MT5 Bridge v3 - Connects MetaTrader 5 to Sovereign v4 Rust engine
TCP server that streams ticks/candles and executes trades
"""

import socket
import json
import time
import threading
from datetime import datetime
import MetaTrader5 as mt5
import pytz

# Configuration
HOST = '0.0.0.0'
PORT = 5555
SYMBOL = "XAUUSD"  # Not XAUUSD.x for demo account
MT5_PATH = r"C:\MT5_Sovereign\terminal64.exe"
CANDLE_TIMEFRAME = mt5.TIMEFRAME_M5

# Timezone
UTC = pytz.UTC

class MT5Bridge:
    def __init__(self):
        self.running = False
        self.client_socket = None
        self.last_candle_time = None
        self.positions = {}
        
    def init_mt5(self):
        """Initialize MT5 connection"""
        if not mt5.initialize(path=MT5_PATH):
            print(f"MT5 init failed: {mt5.last_error()}")
            return False
        
        account_info = mt5.account_info()
        if account_info is None:
            print("Failed to get account info")
            return False
            
        print(f"Connected to MT5: Account {account_info.login}, Balance: ${account_info.balance}")
        
        # Check symbol
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            print(f"Symbol {SYMBOL} not found")
            return False
        if not symbol_info.visible:
            mt5.symbol_select(SYMBOL, True)
            
        print(f"Symbol {SYMBOL} ready, Bid: {symbol_info.bid}, Ask: {symbol_info.ask}")
        return True
    
    def get_tick(self):
        """Get current tick data"""
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            return None
        return {
            "type": "tick",
            "symbol": SYMBOL,
            "bid": tick.bid,
            "ask": tick.ask,
            "time": tick.time,
            "volume": tick.volume
        }
    
    def get_candle(self):
        """Get latest completed candle"""
        candles = mt5.copy_rates_from_pos(SYMBOL, CANDLE_TIMEFRAME, 0, 2)
        if candles is None or len(candles) < 2:
            return None
        
        # Return the completed candle (index -2), not the forming one
        c = candles[-2]
        candle_time = c['time']
        
        # Only send if new candle
        if self.last_candle_time == candle_time:
            return None
            
        self.last_candle_time = candle_time
        
        return {
            "type": "candle",
            "symbol": SYMBOL,
            "timeframe": "M5",
            "time": int(candle_time),
            "open": float(c['open']),
            "high": float(c['high']),
            "low": float(c['low']),
            "close": float(c['close']),
            "volume": float(c['tick_volume'])
        }
    
    def get_account_info(self):
        """Get account information"""
        info = mt5.account_info()
        if info is None:
            return None
        return {
            "type": "account",
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "profit": info.profit
        }
    
    def get_positions(self):
        """Get open positions"""
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions is None:
            return {"type": "positions", "positions": []}
        
        pos_list = []
        for p in positions:
            pos_list.append({
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "buy" if p.type == 0 else "sell",
                "volume": p.volume,
                "open_price": p.price_open,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
                "open_time": p.time
            })
        
        return {"type": "positions", "positions": pos_list}
    
    def execute_order(self, order_data):
        """Execute a trade order"""
        try:
            symbol = order_data.get("symbol", SYMBOL)
            action = order_data.get("action", "").upper()
            volume = float(order_data.get("volume", 0.01))
            sl = order_data.get("sl", 0.0)
            tp = order_data.get("tp", 0.0)
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {"type": "order_result", "success": False, "error": "Symbol not found"}
            
            point = symbol_info.point
            price = mt5.symbol_info_tick(symbol).ask if action == "BUY" else mt5.symbol_info_tick(symbol).bid
            
            if action == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
            elif action == "SELL":
                order_type = mt5.ORDER_TYPE_SELL
            else:
                return {"type": "order_result", "success": False, "error": f"Unknown action: {action}"}
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 123456,
                "comment": "Sovereign v4",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    "type": "order_result",
                    "success": False,
                    "error": f"Order failed: {result.retcode} - {result.comment}"
                }
            
            return {
                "type": "order_result",
                "success": True,
                "ticket": result.order,
                "price": result.price,
                "volume": result.volume
            }
            
        except Exception as e:
            return {"type": "order_result", "success": False, "error": str(e)}
    
    def close_position(self, close_data):
        """Close a position by ticket"""
        try:
            ticket = int(close_data.get("ticket", 0))
            
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {"type": "close_result", "success": False, "error": "Position not found"}
            
            position = position[0]
            symbol = position.symbol
            volume = position.volume
            
            # Reverse the position type
            if position.type == 0:  # Buy position, close with sell
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            else:  # Sell position, close with buy
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": 123456,
                "comment": "Sovereign v4 close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    "type": "close_result",
                    "success": False,
                    "error": f"Close failed: {result.retcode} - {result.comment}"
                }
            
            return {
                "type": "close_result",
                "success": True,
                "ticket": ticket,
                "close_price": result.price,
                "profit": position.profit
            }
            
        except Exception as e:
            return {"type": "close_result", "success": False, "error": str(e)}
    
    def handle_command(self, data):
        """Handle incoming command from Rust"""
        try:
            cmd = json.loads(data)
            cmd_type = cmd.get("cmd", "")
            
            if cmd_type == "order":
                return self.execute_order(cmd)
            elif cmd_type == "close":
                return self.close_position(cmd)
            elif cmd_type == "positions":
                return self.get_positions()
            elif cmd_type == "account":
                return self.get_account_info()
            elif cmd_type == "ping":
                return {"type": "pong", "time": time.time()}
            else:
                return {"type": "error", "error": f"Unknown command: {cmd_type}"}
                
        except json.JSONDecodeError:
            return {"type": "error", "error": "Invalid JSON"}
        except Exception as e:
            return {"type": "error", "error": str(e)}
    
    def send_message(self, msg):
        """Send message to client"""
        if self.client_socket:
            try:
                data = json.dumps(msg) + "\n"
                self.client_socket.send(data.encode())
                return True
            except:
                return False
        return False
    
    def client_receiver(self):
        """Thread to receive commands from client"""
        while self.running and self.client_socket:
            try:
                self.client_socket.settimeout(1.0)
                data = self.client_socket.recv(4096)
                if not data:
                    break
                
                # Handle each line as a command
                for line in data.decode().strip().split('\n'):
                    if line:
                        response = self.handle_command(line)
                        self.send_message(response)
                        
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Receiver error: {e}")
                break
    
    def run(self):
        """Main bridge loop"""
        if not self.init_mt5():
            return
        
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        server.settimeout(5.0)
        
        print(f"MT5 Bridge listening on {HOST}:{PORT}")
        
        self.running = True
        tick_count = 0
        
        while self.running:
            # Wait for client connection
            if self.client_socket is None:
                try:
                    print("Waiting for Sovereign connection...")
                    self.client_socket, addr = server.accept()
                    print(f"Client connected from {addr}")
                    
                    # Start receiver thread
                    receiver = threading.Thread(target=self.client_receiver, daemon=True)
                    receiver.start()
                    
                    # Send initial account info
                    self.send_message(self.get_account_info())
                    self.send_message(self.get_positions())
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Accept error: {e}")
                    continue
            
            # Stream data
            try:
                # Send tick
                tick = self.get_tick()
                if tick:
                    if not self.send_message(tick):
                        print("Client disconnected")
                        self.client_socket = None
                        continue
                    tick_count += 1
                    if tick_count % 100 == 0:
                        print(f"Streamed {tick_count} ticks")
                
                # Check for new candle
                candle = self.get_candle()
                if candle:
                    self.send_message(candle)
                    print(f"Sent candle: {candle['time']} O={candle['open']} H={candle['high']} L={candle['low']} C={candle['close']}")
                
                time.sleep(0.1)  # 10 ticks per second max
                
            except Exception as e:
                print(f"Stream error: {e}")
                self.client_socket = None
                time.sleep(1)
        
        server.close()
        mt5.shutdown()
        print("Bridge shutdown")

if __name__ == "__main__":
    bridge = MT5Bridge()
    try:
        bridge.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
        bridge.running = False