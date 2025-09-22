"""
Test Runner for Order Processing System
Run different scenarios to validate the workflow
"""

import asyncio
import json
from datetime import datetime
from order_processing_system import process_order
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def test_full_stock_scenario():
    """Test scenario where all items are in stock"""
    print("\n" + "=" * 50)
    print("Testing FULL STOCK Scenario")
    print("=" * 50)

    order = {
        "order_id": f"TEST-FULL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "customer_info": {
            "customer_id": "CUST-TEST-001",
            "name": "Alice Johnson",
            "email": "alice@test.com",
            "phone": "+1234567890"
        },
        "items": [
            {
                "product_id": "PROD-A001",
                "name": "Widget A",
                "quantity": 3,
                "price": 29.99
            },
            {
                "product_id": "PROD-B002",
                "name": "Gadget B",
                "quantity": 1,
                "price": 49.99
            }
        ],
        "shipping_address": {
            "street": "456 Test Ave",
            "city": "Test City",
            "state": "TC",
            "zip": "12345"
        },
        "payment_method": "credit_card"
    }

    result = await process_order(order)
    print("\nResult:")
    print(json.dumps(result, indent=2))
    return result


async def test_partial_stock_scenario():
    """Test scenario where some items are out of stock"""
    print("\n" + "=" * 50)
    print("Testing PARTIAL STOCK Scenario")
    print("=" * 50)

    order = {
        "order_id": f"TEST-PARTIAL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "customer_info": {
            "customer_id": "CUST-TEST-002",
            "name": "Bob Smith",
            "email": "bob@test.com",
            "phone": "+9876543210"
        },
        "items": [
            {
                "product_id": "PROD-A001",
                "name": "Widget A",
                "quantity": 2,
                "price": 29.99
            },
            {
                "product_id": "PROD-C003",  # This should be out of stock
                "name": "Device C",
                "quantity": 3,
                "price": 99.99
            }
        ],
        "shipping_address": {
            "street": "789 Partial St",
            "city": "Stock City",
            "state": "SC",
            "zip": "54321"
        },
        "payment_method": "paypal"
    }

    result = await process_order(order)
    print("\nResult:")
    print(json.dumps(result, indent=2))
    return result


async def test_no_stock_scenario():
    """Test scenario where all items are out of stock"""
    print("\n" + "=" * 50)
    print("Testing NO STOCK Scenario")
    print("=" * 50)

    order = {
        "order_id": f"TEST-NOSTOCK-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "customer_info": {
            "customer_id": "CUST-TEST-003",
            "name": "Carol Davis",
            "email": "carol@test.com",
            "phone": "+1122334455"
        },
        "items": [
            {
                "product_id": "PROD-C003",  # Out of stock item
                "name": "Device C",
                "quantity": 10,
                "price": 99.99
            }
        ],
        "shipping_address": {
            "street": "321 Empty Ave",
            "city": "No Stock Town",
            "state": "NS",
            "zip": "00000"
        },
        "payment_method": "bank_transfer"
    }

    result = await process_order(order)
    print("\nResult:")
    print(json.dumps(result, indent=2))
    return result


async def test_qa_failure_scenario():
    """Test scenario with QA failure simulation"""
    print("\n" + "=" * 50)
    print("Testing QA FAILURE Scenario")
    print("=" * 50)

    # This will trigger the supplier flow and QA inspection
    order = {
        "order_id": f"TEST-QA-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "customer_info": {
            "customer_id": "CUST-TEST-004",
            "name": "David QA",
            "email": "david@qatest.com",
            "phone": "+9998887777"
        },
        "items": [
            {
                "product_id": "PROD-D004",
                "name": "Tool D",
                "quantity": 30,  # More than available, will trigger purchase
                "price": 149.99
            }
        ],
        "shipping_address": {
            "street": "QA Test Street",
            "city": "Quality City",
            "state": "QC",
            "zip": "11111"
        },
        "payment_method": "credit_card"
    }

    result = await process_order(order)
    print("\nResult:")
    print(json.dumps(result, indent=2))
    return result


async def run_all_tests():
    """Run all test scenarios"""
    print("\n" + "#" * 60)
    print("# ORDER PROCESSING SYSTEM - COMPREHENSIVE TEST SUITE")
    print("#" * 60)

    # Collect results
    results = {}

    # Run tests sequentially to avoid conflicts
    print("\n🚀 Starting test suite...")

    try:
        # Test 1: Full Stock
        results['full_stock'] = await test_full_stock_scenario()
        await asyncio.sleep(2)  # Brief pause between tests

        # Test 2: Partial Stock
        results['partial_stock'] = await test_partial_stock_scenario()
        await asyncio.sleep(2)

        # Test 3: No Stock
        results['no_stock'] = await test_no_stock_scenario()
        await asyncio.sleep(2)

        # Test 4: QA Failure
        results['qa_failure'] = await test_qa_failure_scenario()

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {str(e)}")
        return

    # Summary Report
    print("\n" + "#" * 60)
    print("# TEST SUITE SUMMARY")
    print("#" * 60)

    for scenario, result in results.items():
        status = "✅ PASSED" if result.get('success') else "❌ FAILED"
        print(f"\n{scenario.upper()}: {status}")
        if result.get('success'):
            print(f"  - Order ID: {result.get('order_id')}")
            print(f"  - Final Status: {result.get('status')}")
            print(f"  - Inventory Status: {result.get('inventory_status')}")
            if result.get('qa_results'):
                print(f"  - QA Results: {len(result.get('qa_results', []))} inspections")
        else:
            print(f"  - Error: {result.get('error')}")

    print("\n" + "#" * 60)
    print("# TEST SUITE COMPLETED")
    print("#" * 60)


async def run_interactive_test():
    """Interactive test mode where user can input custom orders"""
    print("\n" + "=" * 50)
    print("INTERACTIVE ORDER TEST MODE")
    print("=" * 50)

    print("\nEnter order details (or 'quit' to exit):")

    while True:
        print("\n1. Enter customer name: ", end="")
        customer_name = input().strip()
        if customer_name.lower() == 'quit':
            break

        print("2. Enter product ID (e.g., PROD-A001): ", end="")
        product_id = input().strip()

        print("3. Enter quantity: ", end="")
        try:
            quantity = int(input().strip())
        except ValueError:
            print("Invalid quantity. Using 1.")
            quantity = 1

        order = {
            "order_id": f"INTERACTIVE-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "customer_info": {
                "customer_id": f"CUST-INT-{datetime.now().strftime('%H%M%S')}",
                "name": customer_name,
                "email": f"{customer_name.lower().replace(' ', '.')}@test.com"
            },
            "items": [
                {
                    "product_id": product_id,
                    "name": f"Product {product_id}",
                    "quantity": quantity,
                    "price": 99.99
                }
            ],
            "payment_method": "credit_card"
        }

        print("\nProcessing order...")
        result = await process_order(order)

        print("\nResult:")
        print(json.dumps(result, indent=2))

        print("\nTest another order? (yes/no): ", end="")
        if input().strip().lower() != 'yes':
            break


def main():
    """Main entry point"""
    print("\nORDER PROCESSING SYSTEM TEST RUNNER")
    print("Select test mode:")
    print("1. Run all test scenarios")
    print("2. Test full stock scenario only")
    print("3. Test partial stock scenario only")
    print("4. Test no stock scenario only")
    print("5. Test QA failure scenario only")
    print("6. Interactive test mode")
    print("\nEnter choice (1-6): ", end="")

    choice = input().strip()

    if choice == '1':
        asyncio.run(run_all_tests())
    elif choice == '2':
        asyncio.run(test_full_stock_scenario())
    elif choice == '3':
        asyncio.run(test_partial_stock_scenario())
    elif choice == '4':
        asyncio.run(test_no_stock_scenario())
    elif choice == '5':
        asyncio.run(test_qa_failure_scenario())
    elif choice == '6':
        asyncio.run(run_interactive_test())
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()