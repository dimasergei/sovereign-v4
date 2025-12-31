#!/usr/bin/env python3
"""
MT5 Bridge v3 - Connects MetaTrader 5 to Sovereign v4 Rust engine
Text-based protocol over TCP

Protocol FROM bridge TO Rust:
    TICK:bid,ask
    CANDLE:open,high,low,close,volume
    ACCOUNT:balance,equity,profit
    ORDER_OK:ticket,price
    ORDER_ERROR:code,message
    POSITION_OPEN:ticket,side,volume,open_price,sl,tp,profit
    POSITION_UPDATE:ticket,profit
    POSITION_CLOSED
    CLOSE_OK:ticket,profit
    CLOSE_ERROR:message

Protocol FROM Rust TO bridge:
    ACCOUNT
    POSITIONS
    BUY:lots,sl,tp
    SELL:lots,sl,tp
    CLOSE:ticket
"""

import socket
import time
import threading
from datetime import datetime
import MetaTrader5 as mt5

# Configuration
HOST = '0.0.0.0'
PORT = 5555
SYMBOL = "XAUUSD"
MT5_PATH = r"C:\MT5_Sovereign\terminal64.exe"
CANDLE_TIMEFRAME = mt5.TIMEFRAME_M5


class MT5Bridge:
    def __init__(self):
        self.running = False
        self.client_socket = None
        self.last_candle_time = None
        
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
    
    def send(self, message):
        """Send message to Rust client"""
        if self.client_socket:
            try:
                self.client_socket.send((message + "\n").encode())
                return True
            except Exception as e:
                print(f"Send error: {e}")
                return False
        return False
    
    def send_tick(self):
        """Send current tick: TICK:bid,ask"""
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick:
            self.send(f"TICK:{tick.bid},{tick.ask}")
            return True
        return False
    
    def send_candle(self):
        """Send completed candle: CANDLE:open,high,low,close,volume"""
        candles = mt5.copy_rates_from_pos(SYMBOL, CANDLE_TIMEFRAME, 0, 2)
        if candles is None or len(candles) < 2:
            return False
        
        # Get completed candle (not the forming one)
        c = candles[-2]
        candle_time = c['time']
        
        # Only send new candles
        if self.last_candle_time == candle_time:
            return False
            
        self.last_candle_time = candle_time
        msg = f"CANDLE:{c['open']},{c['high']},{c['low']},{c['close']},{c['tick_volume']}"
        self.send(msg)
        print(f"Sent: {msg}")
        return True
    
    def send_account(self):
        """Send account info: ACCOUNT:balance,equity,profit"""
        info = mt5.account_info()
        if info:
            self.send(f"ACCOUNT:{info.balance},{info.equity},{info.profit}")
            print(f"Sent account: balance={info.balance}, equity={info.equity}")
    
    def send_positions(self):
        """Send all open positions"""
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions is None or len(positions) == 0:
            return
        
        for p in positions:
            # POSITION_OPEN:ticket,side,volume,open_price,sl,tp,profit
            side = 0 if p.type == 0 else 1  # 0=buy, 1=sell
            msg = f"POSITION_OPEN:{p.ticket},{side},{p.volume},{p.price_open},{p.sl},{p.tp},{p.profit}"
            self.send(msg)
            print(f"Sent position: ticket={p.ticket}")
    
    def execute_buy(self, lots, sl, tp):
        """Execute BUY order"""
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            self.send("ORDER_ERROR:1,Symbol not found")
            return
        
        price = mt5.symbol_info_tick(SYMBOL).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": float(lots),
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "sl": float(sl) if float(sl) > 0 else 0.0,
            "tp": float(tp) if float(tp) > 0 else 0.0,
            "deviation": 20,
            "magic": 123456,
            "comment": "Sovereign v4",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.send(f"ORDER_OK:{result.order},{result.price}")
            print(f"BUY executed: ticket={result.order}, price={result.price}")
        else:
            self.send(f"ORDER_ERROR:{result.retcode},{result.comment}")
            print(f"BUY failed: {result.retcode} - {result.comment}")
    
    def execute_sell(self, lots, sl, tp):
        """Execute SELL order"""
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            self.send("ORDER_ERROR:1,Symbol not found")
            return
        
        price = mt5.symbol_info_tick(SYMBOL).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": float(lots),
            "type": mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": float(sl) if float(sl) > 0 else 0.0,
            "tp": float(tp) if float(tp) > 0 else 0.0,
            "deviation": 20,
            "magic": 123456,
            "comment": "Sovereign v4",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.send(f"ORDER_OK:{result.order},{result.price}")
            print(f"SELL executed: ticket={result.order}, price={result.price}")
        else:
            self.send(f"ORDER_ERROR:{result.retcode},{result.comment}")
            print(f"SELL failed: {result.retcode} - {result.comment}")
    
    def close_position(self, ticket):
        """Close position by ticket"""
        ticket = int(ticket)
        position = mt5.positions_get(ticket=ticket)
        
        if not position:
            self.send(f"CLOSE_ERROR:Position {ticket} not found")
            return
        
        position = position[0]
        
        # Reverse the position
        if position.type == 0:  # Buy -> close with sell
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(position.symbol).bid
        else:  # Sell -> close with buy
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(position.symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
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
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.send(f"CLOSE_OK:{ticket},{position.profit}")
            self.send("POSITION_CLOSED")
            print(f"Position {ticket} closed, profit={position.profit}")
        else:
            self.send(f"CLOSE_ERROR:{result.comment}")
            print(f"Close failed: {result.retcode} - {result.comment}")
    
    def handle_command(self, cmd):
        """Handle command from Rust"""
        cmd = cmd.strip()
        if not cmd:
            return
        
        print(f"Received: {cmd}")
        
        if cmd == "ACCOUNT":
            self.send_account()
        
        elif cmd == "POSITIONS":
            self.send_positions()
        
        elif cmd.startswith("BUY:"):
            parts = cmd[4:].split(',')
            if len(parts) >= 3:
                self.execute_buy(parts[0], parts[1], parts[2])
        
        elif cmd.startswith("SELL:"):
            parts = cmd[5:].split(',')
            if len(parts) >= 3:
                self.execute_sell(parts[0], parts[1], parts[2])
        
        elif cmd.startswith("CLOSE:"):
            ticket = cmd[6:]
            self.close_position(ticket)
    
    def receiver_thread(self):
        """Thread to receive commands from Rust"""
        while self.running and self.client_socket:
            try:
                self.client_socket.settimeout(1.0)
                data = self.client_socket.recv(4096)
                if not data:
                    break
                
                for line in data.decode().strip().split('\n'):
                    self.handle_command(line)
                    
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Receiver error: {e}")
                break
    
    def run(self):
        """Main loop"""
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
            # Wait for connection
            if self.client_socket is None:
                try:
                    print("Waiting for Sovereign connection...")
                    self.client_socket, addr = server.accept()
                    print(f"Client connected from {addr}")
                    
                    # Start receiver thread
                    receiver = threading.Thread(target=self.receiver_thread, daemon=True)
                    receiver.start()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Accept error: {e}")
                    continue
            
            # Stream data
            try:
                # Send tick
                if self.send_tick():
                    tick_count += 1
                    if tick_count % 100 == 0:
                        print(f"Streamed {tick_count} ticks")
                else:
                    # send_tick failed, client disconnected
                    print("Client disconnected")
                    self.client_socket = None
                    tick_count = 0
                    continue
                
                # Check for new candle
                self.send_candle()
                
                time.sleep(0.1)  # 10 ticks/sec max
                
            except Exception as e:
                print(f"Stream error: {e}")
                self.client_socket = None
                tick_count = 0
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