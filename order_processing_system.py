"""
Complete LangGraph Order Processing System with Supabase and Anthropic Integration
"""

import os
import json
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
from enum import Enum
from datetime import datetime
import asyncio
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from supabase import create_client, Client
import logging
from dotenv import load_dotenv


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================== Configuration ==================
# Load the .env file
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================== State Definitions ==================
class InventoryStatus(str, Enum):
    FULL_STOCK = "full_stock"
    PARTIAL_STOCK = "partial_stock"
    NO_STOCK = "no_stock"

class OrderState(TypedDict):
    """Main state for the order processing workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    order_id: str
    customer_info: Dict[str, Any]
    order_items: List[Dict[str, Any]]
    inventory_status: Optional[InventoryStatus]
    inventory_details: Dict[str, Any]
    payment_status: Optional[str]
    payment_records: List[Dict[str, Any]]
    delivery_status: Optional[str]
    tracking_info: Dict[str, Any]
    purchase_requests: List[Dict[str, Any]]
    qa_results: List[Dict[str, Any]]
    qa_pass_count: int
    qa_fail_count: int
    treatment_decision: Optional[str]
    final_status: Optional[str]
    current_step: str
    error_log: List[Dict[str, Any]]
    metadata: Dict[str, Any]

# ================== Agent Prompts ==================
ADMIN_AGENT_PROMPT = """You are an Admin Agent responsible for managing order processing workflows.
Your responsibilities:
1. Extract order information from customer conversations
2. Validate order details and customer information
3. Check inventory status and make routing decisions
4. Coordinate with other agents for order fulfillment
5. Handle customer communications professionally

When receiving an order:
- Extract all relevant order details (items, quantities, customer info)
- Verify completeness of information
- Check against available inventory
- Route to appropriate next steps based on stock availability

Be precise, professional, and customer-focused."""

DATABASE_AGENT_PROMPT = """You are a Database Agent responsible for all database operations.
Your responsibilities:
1. Query and update order records in Supabase
2. Maintain accurate inventory counts
3. Record all transaction states
4. Generate reports when needed
5. Ensure data integrity and consistency

Always:
- Validate data before inserting/updating
- Log all database operations
- Handle errors gracefully
- Maintain audit trails"""

FINANCE_AGENT_PROMPT = """You are a Finance Agent responsible for payment and financial operations.
Your responsibilities:
1. Process payment verifications
2. Generate and send invoices
3. Track payment status
4. Handle refunds and adjustments
5. Maintain financial records

Always:
- Verify payment details thoroughly
- Ensure compliance with financial regulations
- Keep accurate records
- Coordinate with other agents on financial matters"""

OPERATION_AGENT_PROMPT = """You are an Operation Agent responsible for fulfillment and logistics.
Your responsibilities:
1. Coordinate order fulfillment
2. Manage supplier relationships
3. Track shipments and deliveries
4. Handle purchase requests for out-of-stock items
5. Ensure operational efficiency

Focus on:
- Timely order processing
- Accurate tracking information
- Supplier coordination
- Inventory replenishment"""

QA_AGENT_PROMPT = """You are a QA Agent responsible for quality assurance.
Your responsibilities:
1. Verify order accuracy
2. Check product quality (when applicable)
3. Validate fulfillment processes
4. Make pass/fail decisions
5. Recommend corrective actions

Evaluation criteria:
- Order completeness
- Data accuracy
- Process compliance
- Customer satisfaction potential"""

# ================== Agent Classes ==================
class BaseAgent:
    """Base class for all agents"""
    def __init__(self, name: str, prompt: str):
        self.name = name
        self.prompt = prompt
        self.llm = ChatAnthropic(
            model="claude-3-haiku-20240307",
            api_key=ANTHROPIC_API_KEY,
            temperature=0.3
        )

    async def process(self, state: OrderState, specific_task: str = "") -> Dict[str, Any]:
        """Process the current state and return updates"""
        try:
            messages = [
                SystemMessage(content=self.prompt),
                HumanMessage(content=f"""
                Current State:
                - Order ID: {state.get('order_id', 'Not set')}
                - Current Step: {state.get('current_step', 'Unknown')}
                - Inventory Status: {state.get('inventory_status', 'Not checked')}
                
                Specific Task: {specific_task}
                
                Order Details: {json.dumps(state.get('order_items', []), indent=2)}
                Customer Info: {json.dumps(state.get('customer_info', {}), indent=2)}
                
                Please process this information and provide your response.
                """)
            ]

            response = await self.llm.ainvoke(messages)

            return {
                "agent": self.name,
                "response": response.content,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")
            return {
                "agent": self.name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# ================== Specific Agent Implementations ==================
class AdminAgent(BaseAgent):
    def __init__(self):
        super().__init__("Admin Agent", ADMIN_AGENT_PROMPT)

    async def receive_order(self, state: OrderState) -> OrderState:
        """Initial order reception and processing"""
        logger.info("Admin Agent: Receiving order")

        # Extract order information from messages
        result = await self.process(state, "Extract and validate order information from the conversation")

        # Update state with extracted information
        state['current_step'] = 'order_received'
        state['messages'].append(HumanMessage(content=f"Admin Agent processed: {result['response']}"))

        # Set sample order data (in production, this would be extracted from the conversation)
        if not state.get('order_id'):
            state['order_id'] = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        if not state.get('order_items'):
            state['order_items'] = [
                {"product_id": "PROD-001", "quantity": 2, "name": "Sample Product"}
            ]

        if not state.get('customer_info'):
            state['customer_info'] = {
                "customer_id": "CUST-001",
                "name": "John Doe",
                "email": "john@example.com"
            }

        return state

    async def check_inventory(self, state: OrderState) -> OrderState:
        """Check inventory status for order items"""
        logger.info("Admin Agent: Checking inventory")

        # Query inventory from database
        try:
            inventory_data = {}
            for item in state['order_items']:
                # Query Supabase for inventory
                response = supabase.table('inventory').select('*').eq('product_id', item['product_id']).execute()

                if response.data:
                    inventory_data[item['product_id']] = {
                        'available': response.data[0].get('quantity', 0),
                        'requested': item['quantity']
                    }
                else:
                    inventory_data[item['product_id']] = {
                        'available': 0,
                        'requested': item['quantity']
                    }

            state['inventory_details'] = inventory_data

            # Determine overall inventory status
            all_available = all(
                inv['available'] >= inv['requested']
                for inv in inventory_data.values()
            )
            none_available = all(
                inv['available'] == 0
                for inv in inventory_data.values()
            )

            if all_available:
                state['inventory_status'] = InventoryStatus.FULL_STOCK
            elif none_available:
                state['inventory_status'] = InventoryStatus.NO_STOCK
            else:
                state['inventory_status'] = InventoryStatus.PARTIAL_STOCK

            state['current_step'] = 'inventory_checked'

        except Exception as e:
            logger.error(f"Inventory check error: {str(e)}")
            state['error_log'].append({"step": "inventory_check", "error": str(e)})

        return state

    async def confirm_customer(self, state: OrderState) -> OrderState:
        """Confirm with customer about order status"""
        logger.info("Admin Agent: Confirming with customer")

        result = await self.process(state, f"Confirm order status with customer. Inventory status: {state['inventory_status']}")
        state['messages'].append(HumanMessage(content=f"Customer confirmation: {result['response']}"))
        state['current_step'] = 'customer_confirmed'

        return state

class DatabaseAgent(BaseAgent):
    def __init__(self):
        super().__init__("Database Agent", DATABASE_AGENT_PROMPT)

    async def process_delivery(self, state: OrderState) -> OrderState:
        """Process delivery and update database"""
        logger.info("Database Agent: Processing delivery")

        try:
            # Update order status in database
            order_update = {
                "order_id": state['order_id'],
                "status": "processing_delivery",
                "inventory_status": state['inventory_status'].value,
                "updated_at": datetime.now().isoformat()
            }

            response = supabase.table('orders').upsert(order_update).execute()

            state['current_step'] = 'delivery_processed'
            state['delivery_status'] = 'initiated'

        except Exception as e:
            logger.error(f"Database update error: {str(e)}")
            state['error_log'].append({"step": "database_delivery", "error": str(e)})

        return state

    async def deduct_inventory(self, state: OrderState) -> OrderState:
        """Deduct inventory from database"""
        logger.info("Database Agent: Deducting inventory")

        try:
            for item in state['order_items']:
                # Get current inventory
                response = supabase.table('inventory').select('*').eq('product_id', item['product_id']).execute()

                if response.data:
                    current_qty = response.data[0]['quantity']
                    new_qty = max(0, current_qty - item['quantity'])

                    # Update inventory
                    supabase.table('inventory').update({
                        'quantity': new_qty,
                        'last_updated': datetime.now().isoformat()
                    }).eq('product_id', item['product_id']).execute()

            state['current_step'] = 'inventory_deducted'

        except Exception as e:
            logger.error(f"Inventory deduction error: {str(e)}")
            state['error_log'].append({"step": "inventory_deduction", "error": str(e)})

        return state

    async def update_payment_records(self, state: OrderState) -> OrderState:
        """Update payment records in database"""
        logger.info("Database Agent: Updating payment records")

        try:
            payment_record = {
                "order_id": state['order_id'],
                "amount": sum(item.get('price', 100) * item['quantity'] for item in state['order_items']),
                "status": state.get('payment_status', 'pending'),
                "created_at": datetime.now().isoformat()
            }

            response = supabase.table('payments').insert(payment_record).execute()

            state['payment_records'].append(payment_record)
            state['current_step'] = 'payment_recorded'

        except Exception as e:
            logger.error(f"Payment record error: {str(e)}")
            state['error_log'].append({"step": "payment_recording", "error": str(e)})

        return state

class FinanceAgent(BaseAgent):
    def __init__(self):
        super().__init__("Finance Agent", FINANCE_AGENT_PROMPT)

    async def review_purchase_request(self, state: OrderState) -> OrderState:
        """Review and approve purchase request"""
        logger.info("Finance Agent: Reviewing purchase request")

        result = await self.process(state, "Review purchase request for out-of-stock items")

        # Simulate approval
        state['purchase_requests'].append({
            "request_id": f"PR-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "status": "approved",
            "items": state['order_items'],
            "approved_by": "Finance Agent",
            "timestamp": datetime.now().isoformat()
        })

        state['current_step'] = 'purchase_approved'
        state['messages'].append(HumanMessage(content=f"Finance review: {result['response']}"))

        return state

    async def generate_invoice(self, state: OrderState) -> OrderState:
        """Generate invoice for the order"""
        logger.info("Finance Agent: Generating invoice")

        invoice = {
            "invoice_id": f"INV-{state['order_id']}",
            "order_id": state['order_id'],
            "customer": state['customer_info'],
            "items": state['order_items'],
            "total": sum(item.get('price', 100) * item['quantity'] for item in state['order_items']),
            "created_at": datetime.now().isoformat()
        }

        state['metadata']['invoice'] = invoice
        state['current_step'] = 'invoice_generated'

        return state

    async def track_payment(self, state: OrderState) -> OrderState:
        """Track payment status"""
        logger.info("Finance Agent: Tracking payment")

        # Check payment status from database
        try:
            response = supabase.table('payments').select('*').eq('order_id', state['order_id']).execute()

            if response.data:
                state['payment_status'] = response.data[0].get('status', 'pending')
            else:
                state['payment_status'] = 'not_found'

            state['current_step'] = 'payment_tracked'

        except Exception as e:
            logger.error(f"Payment tracking error: {str(e)}")
            state['error_log'].append({"step": "payment_tracking", "error": str(e)})

        return state

class OperationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Operation Agent", OPERATION_AGENT_PROMPT)

    async def create_purchase_request(self, state: OrderState) -> OrderState:
        """Create purchase request for out-of-stock items"""
        logger.info("Operation Agent: Creating purchase request")

        out_of_stock_items = []
        for item in state['order_items']:
            inv_detail = state['inventory_details'].get(item['product_id'], {})
            if inv_detail.get('available', 0) < item['quantity']:
                out_of_stock_items.append({
                    **item,
                    'shortage': item['quantity'] - inv_detail.get('available', 0)
                })

        if out_of_stock_items:
            purchase_request = {
                "request_id": f"PR-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "order_id": state['order_id'],
                "items": out_of_stock_items,
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }

            state['purchase_requests'].append(purchase_request)

        state['current_step'] = 'purchase_request_created'

        return state

    async def send_to_supplier(self, state: OrderState) -> OrderState:
        """Send purchase request to supplier"""
        logger.info("Operation Agent: Sending to supplier")

        result = await self.process(state, "Send approved purchase request to supplier")

        # Update purchase request status
        if state['purchase_requests']:
            state['purchase_requests'][-1]['status'] = 'sent_to_supplier'
            state['purchase_requests'][-1]['sent_at'] = datetime.now().isoformat()

        state['current_step'] = 'sent_to_supplier'
        state['messages'].append(HumanMessage(content=f"Supplier notification: {result['response']}"))

        return state

    async def coordinate_with_supplier(self, state: OrderState) -> OrderState:
        """Coordinate delivery with supplier"""
        logger.info("Operation Agent: Coordinating with supplier")

        result = await self.process(state, "Coordinate delivery timeline and details with supplier")

        state['tracking_info'] = {
            "supplier": "Default Supplier",
            "estimated_arrival": "3-5 business days",
            "tracking_number": f"TRK-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "status": "in_transit"
        }

        state['current_step'] = 'supplier_coordinated'
        state['messages'].append(HumanMessage(content=f"Supplier coordination: {result['response']}"))

        return state

    async def track_parts_arrival(self, state: OrderState) -> OrderState:
        """Track parts arrival from supplier"""
        logger.info("Operation Agent: Tracking parts arrival")

        # Simulate parts arrival check
        state['tracking_info']['status'] = 'arrived'
        state['tracking_info']['arrived_at'] = datetime.now().isoformat()

        state['current_step'] = 'parts_arrived'

        return state

    async def remove_from_supplier(self, state: OrderState) -> OrderState:
        """Remove/close supplier order"""
        logger.info("Operation Agent: Removing supplier order")

        if state['purchase_requests']:
            state['purchase_requests'][-1]['status'] = 'completed'
            state['purchase_requests'][-1]['completed_at'] = datetime.now().isoformat()

        state['current_step'] = 'supplier_order_closed'

        return state

class QAAgent(BaseAgent):
    def __init__(self):
        super().__init__("QA Agent", QA_AGENT_PROMPT)

    async def inspect_received_parts(self, state: OrderState) -> OrderState:
        """Inspect parts received from supplier"""
        logger.info("QA Agent: Inspecting received parts")

        result = await self.process(state, "Inspect quality of received parts from supplier")

        # Simulate QA inspection (random pass/fail for demonstration)
        import random
        qa_passed = random.choice([True, True, False])  # 66% pass rate

        qa_result = {
            "inspection_id": f"QA-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "type": "parts_inspection",
            "result": "pass" if qa_passed else "fail",
            "notes": result['response'],
            "inspector": "QA Agent",
            "timestamp": datetime.now().isoformat()
        }

        state['qa_results'].append(qa_result)

        if qa_passed:
            state['qa_pass_count'] += 1
            state['current_step'] = 'qa_passed'
        else:
            state['qa_fail_count'] += 1
            state['current_step'] = 'qa_failed'

        return state

    async def create_qa_report(self, state: OrderState) -> OrderState:
        """Create comprehensive QA report"""
        logger.info("QA Agent: Creating QA report")

        qa_report = {
            "report_id": f"QAR-{state['order_id']}",
            "order_id": state['order_id'],
            "total_inspections": len(state['qa_results']),
            "pass_count": state['qa_pass_count'],
            "fail_count": state['qa_fail_count'],
            "results": state['qa_results'],
            "created_at": datetime.now().isoformat()
        }

        state['metadata']['qa_report'] = qa_report
        state['current_step'] = 'qa_report_created'

        return state

    async def request_replacement(self, state: OrderState) -> OrderState:
        """Request replacement for failed QA items"""
        logger.info("QA Agent: Requesting replacement")

        replacement_request = {
            "request_id": f"REPL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "order_id": state['order_id'],
            "reason": "QA failure",
            "qa_results": state['qa_results'][-1] if state['qa_results'] else None,
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }

        state['metadata']['replacement_request'] = replacement_request
        state['current_step'] = 'replacement_requested'

        return state

    async def schedule_repair(self, state: OrderState) -> OrderState:
        """Schedule repair for items"""
        logger.info("QA Agent: Scheduling repair")

        repair_schedule = {
            "schedule_id": f"REP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "order_id": state['order_id'],
            "scheduled_date": "Next available slot",
            "estimated_duration": "2-3 business days",
            "status": "scheduled",
            "created_at": datetime.now().isoformat()
        }

        state['metadata']['repair_schedule'] = repair_schedule
        state['current_step'] = 'repair_scheduled'

        return state

    async def track_repair_progress(self, state: OrderState) -> OrderState:
        """Track repair progress"""
        logger.info("QA Agent: Tracking repair progress")

        if 'repair_schedule' in state['metadata']:
            state['metadata']['repair_schedule']['status'] = 'completed'
            state['metadata']['repair_schedule']['completed_at'] = datetime.now().isoformat()

        state['current_step'] = 'repair_completed'

        return state

    async def handle_replacement_process(self, state: OrderState) -> OrderState:
        """Handle the replacement process"""
        logger.info("QA Agent: Handling replacement process")

        if 'replacement_request' in state['metadata']:
            state['metadata']['replacement_request']['status'] = 'processing'
            state['metadata']['replacement_request']['processed_at'] = datetime.now().isoformat()

        state['current_step'] = 'replacement_processing'

        return state

    async def update_replacement_status(self, state: OrderState) -> OrderState:
        """Update replacement status"""
        logger.info("QA Agent: Updating replacement status")

        if 'replacement_request' in state['metadata']:
            state['metadata']['replacement_request']['status'] = 'completed'
            state['metadata']['replacement_request']['completed_at'] = datetime.now().isoformat()

        state['current_step'] = 'replacement_completed'

        return state

    async def make_treatment_decision(self, state: OrderState) -> OrderState:
        """Make treatment decision based on QA results"""
        logger.info("QA Agent: Making treatment decision")

        # Analyze QA results to make decision
        if state['qa_pass_count'] > state['qa_fail_count']:
            state['treatment_decision'] = 'proceed_with_order'
        elif state['qa_fail_count'] > 0 and state['qa_fail_count'] <= 2:
            state['treatment_decision'] = 'repair_required'
        else:
            state['treatment_decision'] = 'full_replacement'

        state['current_step'] = 'treatment_decided'

        return state

# ================== Node Functions ==================
async def start_node(state: OrderState) -> OrderState:
    """Initial node"""
    logger.info("Starting order processing workflow")
    state['current_step'] = 'started'
    state['messages'] = []
    state['error_log'] = []
    state['payment_records'] = []
    state['purchase_requests'] = []
    state['qa_results'] = []
    state['qa_pass_count'] = 0
    state['qa_fail_count'] = 0
    state['metadata'] = {}
    return state

async def admin_receive_order_node(state: OrderState) -> OrderState:
    """Admin agent receives order"""
    agent = AdminAgent()
    return await agent.receive_order(state)

async def admin_check_inventory_node(state: OrderState) -> OrderState:
    """Admin agent checks inventory"""
    agent = AdminAgent()
    return await agent.check_inventory(state)

async def admin_process_delivery_node(state: OrderState) -> OrderState:
    """Admin agent processes delivery for full stock"""
    agent = AdminAgent()
    return await agent.confirm_customer(state)

async def admin_process_partial_node(state: OrderState) -> OrderState:
    """Admin agent processes partial stock"""
    agent = AdminAgent()
    return await agent.confirm_customer(state)

async def operation_create_purchase_node(state: OrderState) -> OrderState:
    """Operation agent creates purchase request"""
    agent = OperationAgent()
    return await agent.create_purchase_request(state)

async def finance_review_purchase_node(state: OrderState) -> OrderState:
    """Finance agent reviews purchase request"""
    agent = FinanceAgent()
    return await agent.review_purchase_request(state)

async def database_deduct_inventory_node(state: OrderState) -> OrderState:
    """Database agent deducts inventory"""
    agent = DatabaseAgent()
    return await agent.deduct_inventory(state)

async def finance_generate_invoice_node(state: OrderState) -> OrderState:
    """Finance agent generates invoice"""
    agent = FinanceAgent()
    return await agent.generate_invoice(state)

async def finance_track_payment_node(state: OrderState) -> OrderState:
    """Finance agent tracks payment"""
    agent = FinanceAgent()
    return await agent.track_payment(state)

async def database_update_payment_node(state: OrderState) -> OrderState:
    """Database agent updates payment records"""
    agent = DatabaseAgent()
    return await agent.update_payment_records(state)

async def admin_send_delivery_node(state: OrderState) -> OrderState:
    """Admin agent sends delivery reminder"""
    logger.info("Admin Agent: Sending delivery reminder")
    state['current_step'] = 'delivery_reminder_sent'
    return state

async def finance_create_purchase_node(state: OrderState) -> OrderState:
    """Finance agent creates purchase request for no stock"""
    agent = FinanceAgent()
    return await agent.review_purchase_request(state)

async def operation_send_to_supplier_node(state: OrderState) -> OrderState:
    """Operation agent sends to supplier"""
    agent = OperationAgent()
    return await agent.send_to_supplier(state)

async def operation_coordinate_supplier_node(state: OrderState) -> OrderState:
    """Operation agent coordinates with supplier"""
    agent = OperationAgent()
    return await agent.coordinate_with_supplier(state)

async def database_update_arrival_node(state: OrderState) -> OrderState:
    """Database agent updates expected arrival"""
    logger.info("Database Agent: Updating expected arrival")
    state['current_step'] = 'arrival_updated'
    return state

async def operation_track_arrival_node(state: OrderState) -> OrderState:
    """Operation agent tracks parts arrival"""
    agent = OperationAgent()
    return await agent.track_parts_arrival(state)

async def operation_remove_supplier_node(state: OrderState) -> OrderState:
    """Operation agent removes supplier order"""
    agent = OperationAgent()
    return await agent.remove_from_supplier(state)

async def qa_inspect_parts_node(state: OrderState) -> OrderState:
    """QA agent inspects received parts"""
    agent = QAAgent()
    return await agent.inspect_received_parts(state)

async def qa_create_report_node(state: OrderState) -> OrderState:
    """QA agent creates report"""
    agent = QAAgent()
    return await agent.create_qa_report(state)

async def qa_request_replacement_node(state: OrderState) -> OrderState:
    """QA agent requests replacement"""
    agent = QAAgent()
    return await agent.request_replacement(state)

async def operation_handle_replacement_node(state: OrderState) -> OrderState:
    """Operation agent handles replacement process"""
    agent = QAAgent()
    return await agent.handle_replacement_process(state)

async def database_update_replacement_node(state: OrderState) -> OrderState:
    """Database agent updates replacement status"""
    agent = QAAgent()
    return await agent.update_replacement_status(state)

async def qa_schedule_repair_node(state: OrderState) -> OrderState:
    """QA agent schedules repair"""
    agent = QAAgent()
    return await agent.schedule_repair(state)

async def qa_track_repair_node(state: OrderState) -> OrderState:
    """QA agent tracks repair progress"""
    agent = QAAgent()
    return await agent.track_repair_progress(state)

async def qa_final_inspection_node(state: OrderState) -> OrderState:
    """QA agent performs final inspection"""
    agent = QAAgent()
    await agent.make_treatment_decision(state)
    return state

async def database_update_inventory_node(state: OrderState) -> OrderState:
    """Database agent updates inventory after QA"""
    logger.info("Database Agent: Updating inventory post-QA")
    state['current_step'] = 'final_inventory_updated'
    state['final_status'] = 'completed'
    return state

async def end_node(state: OrderState) -> OrderState:
    """Final node"""
    logger.info(f"Order {state['order_id']} processing completed")
    state['current_step'] = 'completed'
    state['final_status'] = 'success'
    return state

# ================== Conditional Edge Functions ==================
def check_inventory_status(state: OrderState) -> str:
    """Determine next step based on inventory status"""
    status = state.get('inventory_status')

    if status == InventoryStatus.FULL_STOCK:
        return "full_stock"
    elif status == InventoryStatus.PARTIAL_STOCK:
        return "partial_stock"
    elif status == InventoryStatus.NO_STOCK:
        return "no_stock"
    else:
        return "no_stock"  # Default

def check_customer_confirmation(state: OrderState) -> str:
    """Check if customer confirmed the order"""
    # For demonstration, always return confirmed
    return "confirmed"

def check_purchase_approval(state: OrderState) -> str:
    """Check if purchase request is approved"""
    if state['purchase_requests'] and state['purchase_requests'][-1]['status'] == 'approved':
        return "approved"
    return "rejected"

def check_parts_arrival(state: OrderState) -> str:
    """Check if parts have arrived"""
    if state.get('tracking_info', {}).get('status') == 'arrived':
        return "arrived"
    return "pending"

def check_qa_result(state: OrderState) -> str:
    """Check QA inspection result"""
    if state['qa_results'] and state['qa_results'][-1]['result'] == 'pass':
        return "pass"
    return "fail"

def check_treatment_decision(state: OrderState) -> str:
    """Check treatment decision"""
    decision = state.get('treatment_decision', 'proceed_with_order')

    if decision == 'repair_required':
        return "repair"
    elif decision == 'full_replacement':
        return "replacement"
    else:
        return "proceed"

def check_final_qa_pass(state: OrderState) -> str:
    """Check if final QA passed after treatment"""
    # For demonstration, always pass after treatment
    return "pass"

# ================== Build Graph ==================
def build_order_processing_graph():
    """Build the complete order processing graph"""

    # Initialize the graph
    graph = StateGraph(OrderState)

    # Add nodes
    graph.add_node("start", start_node)
    graph.add_node("admin_receive_order", admin_receive_order_node)
    graph.add_node("admin_check_inventory", admin_check_inventory_node)

    # Full stock path
    graph.add_node("admin_process_delivery_full", admin_process_delivery_node)
    graph.add_node("admin_confirm_customer_full", admin_process_delivery_node)
    graph.add_node("database_deduct_inventory", database_deduct_inventory_node)
    graph.add_node("finance_generate_invoice", finance_generate_invoice_node)
    graph.add_node("finance_track_payment", finance_track_payment_node)
    graph.add_node("database_update_payment", database_update_payment_node)

    # Partial stock path
    graph.add_node("admin_process_partial", admin_process_partial_node)
    graph.add_node("admin_confirm_customer_partial", admin_process_partial_node)
    graph.add_node("admin_send_delivery_reminder", admin_send_delivery_node)

    # No stock path
    graph.add_node("operation_create_purchase", operation_create_purchase_node)
    graph.add_node("finance_review_purchase", finance_review_purchase_node)
    graph.add_node("finance_create_purchase", finance_create_purchase_node)
    graph.add_node("operation_send_to_supplier", operation_send_to_supplier_node)
    graph.add_node("operation_coordinate_supplier", operation_coordinate_supplier_node)
    graph.add_node("database_update_arrival", database_update_arrival_node)
    graph.add_node("operation_track_arrival", operation_track_arrival_node)
    graph.add_node("operation_remove_supplier", operation_remove_supplier_node)

    # QA path
    graph.add_node("qa_inspect_parts", qa_inspect_parts_node)
    graph.add_node("qa_create_report", qa_create_report_node)
    graph.add_node("qa_request_replacement", qa_request_replacement_node)
    graph.add_node("operation_handle_replacement", operation_handle_replacement_node)
    graph.add_node("database_update_replacement", database_update_replacement_node)
    graph.add_node("qa_schedule_repair", qa_schedule_repair_node)
    graph.add_node("qa_track_repair", qa_track_repair_node)
    graph.add_node("qa_final_inspection", qa_final_inspection_node)
    graph.add_node("database_update_inventory_final", database_update_inventory_node)

    graph.add_node("end", end_node)

    # Add edges
    graph.set_entry_point("start")
    graph.add_edge("start", "admin_receive_order")
    graph.add_edge("admin_receive_order", "admin_check_inventory")

    # Conditional routing based on inventory
    graph.add_conditional_edges(
        "admin_check_inventory",
        check_inventory_status,
        {
            "full_stock": "admin_process_delivery_full",
            "partial_stock": "admin_process_partial",
            "no_stock": "operation_create_purchase"
        }
    )

    # Full stock flow
    graph.add_edge("admin_process_delivery_full", "admin_confirm_customer_full")
    graph.add_conditional_edges(
        "admin_confirm_customer_full",
        check_customer_confirmation,
        {
            "confirmed": "database_deduct_inventory",
            "rejected": "end"
        }
    )
    graph.add_edge("database_deduct_inventory", "finance_generate_invoice")
    graph.add_edge("finance_generate_invoice", "finance_track_payment")
    graph.add_edge("finance_track_payment", "database_update_payment")
    graph.add_edge("database_update_payment", "end")

    # Partial stock flow
    graph.add_edge("admin_process_partial", "admin_confirm_customer_partial")
    graph.add_conditional_edges(
        "admin_confirm_customer_partial",
        check_customer_confirmation,
        {
            "confirmed": "admin_send_delivery_reminder",
            "rejected": "end"
        }
    )
    graph.add_edge("admin_send_delivery_reminder", "finance_create_purchase")
    graph.add_edge("finance_create_purchase", "operation_send_to_supplier")

    # No stock flow
    graph.add_edge("operation_create_purchase", "finance_review_purchase")
    graph.add_conditional_edges(
        "finance_review_purchase",
        check_purchase_approval,
        {
            "approved": "operation_send_to_supplier",
            "rejected": "end"
        }
    )

    # Supplier flow
    graph.add_edge("operation_send_to_supplier", "operation_coordinate_supplier")
    graph.add_edge("operation_coordinate_supplier", "database_update_arrival")
    graph.add_edge("database_update_arrival", "operation_track_arrival")

    # Parts arrival conditional
    graph.add_conditional_edges(
        "operation_track_arrival",
        check_parts_arrival,
        {
            "arrived": "operation_remove_supplier",
            "pending": "operation_track_arrival"
        }
    )

    graph.add_edge("operation_remove_supplier", "qa_inspect_parts")

    # QA flow
    graph.add_conditional_edges(
        "qa_inspect_parts",
        check_qa_result,
        {
            "pass": "qa_create_report",
            "fail": "qa_final_inspection"
        }
    )

    graph.add_edge("qa_create_report", "database_update_inventory_final")

    # Treatment decision flow
    graph.add_conditional_edges(
        "qa_final_inspection",
        check_treatment_decision,
        {
            "repair": "qa_schedule_repair",
            "replacement": "qa_request_replacement",
            "proceed": "database_update_inventory_final"
        }
    )

    # Repair flow
    graph.add_edge("qa_schedule_repair", "qa_track_repair")
    graph.add_edge("qa_track_repair", "qa_inspect_parts")

    # Replacement flow
    graph.add_edge("qa_request_replacement", "operation_handle_replacement")
    graph.add_edge("operation_handle_replacement", "database_update_replacement")
    graph.add_edge("database_update_replacement", "qa_inspect_parts")

    # Final
    graph.add_edge("database_update_inventory_final", "end")

    # Compile the graph
    compiled_graph = graph.compile(checkpointer=MemorySaver())

    return compiled_graph

# ================== Main Execution ==================
async def process_order(order_data: Dict[str, Any]):
    """Main function to process an order through the workflow"""

    # Build the graph
    graph = build_order_processing_graph()

    # Initialize state
    initial_state: OrderState = {
        "messages": [HumanMessage(content=f"Processing order: {json.dumps(order_data)}")],
        "order_id": order_data.get("order_id", ""),
        "customer_info": order_data.get("customer_info", {}),
        "order_items": order_data.get("items", []),
        "inventory_status": None,
        "inventory_details": {},
        "payment_status": None,
        "payment_records": [],
        "delivery_status": None,
        "tracking_info": {},
        "purchase_requests": [],
        "qa_results": [],
        "qa_pass_count": 0,
        "qa_fail_count": 0,
        "treatment_decision": None,
        "final_status": None,
        "current_step": "init",
        "error_log": [],
        "metadata": {}
    }

    # Run the graph
    config = {"configurable": {"thread_id": f"order-{order_data.get('order_id', 'unknown')}"}}

    try:
        # Execute the graph
        final_state = await graph.ainvoke(initial_state, config)

        # Log the final state
        logger.info(f"Order processing completed: {final_state['final_status']}")
        logger.info(f"Final step: {final_state['current_step']}")

        return {
            "success": True,
            "order_id": final_state['order_id'],
            "status": final_state['final_status'],
            "inventory_status": final_state['inventory_status'],
            "payment_status": final_state['payment_status'],
            "qa_results": final_state['qa_results'],
            "errors": final_state['error_log']
        }

    except Exception as e:
        logger.error(f"Error processing order: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# ================== Example Usage ==================
if __name__ == "__main__":
    # Example order data
    sample_order = {
        "order_id": "ORD-2024-001",
        "customer_info": {
            "customer_id": "CUST-123",
            "name": "John Smith",
            "email": "john.smith@example.com",
            "phone": "+1234567890"
        },
        "items": [
            {
                "product_id": "PROD-A001",
                "name": "Widget A",
                "quantity": 5,
                "price": 29.99
            },
            {
                "product_id": "PROD-B002",
                "name": "Gadget B",
                "quantity": 2,
                "price": 49.99
            }
        ],
        "shipping_address": {
            "street": "123 Main St",
            "city": "New York",
            "state": "NY",
            "zip": "10001",
            "country": "USA"
        },
        "payment_method": "credit_card"
    }

    # Run the order processing
    import asyncio
    result = asyncio.run(process_order(sample_order))
    print(f"\nOrder Processing Result:")
    print(json.dumps(result, indent=2))